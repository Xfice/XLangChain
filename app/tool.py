"""Reusable LangChain-compatible tool for public X data analysis."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd

from app.processing import clean_text, extract_hashtags, map_sentiment
from app.sources.kaggle_source import fetch_kaggle_dataset_to_csv
from app.sources.playwright_source import fetch_public_x_with_playwright

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "sample.csv"


@dataclass
class TwitterDataTool:
    """Analyze public X/Twitter-like posts from a local CSV dataset."""

    dataset_path: str | Path = os.getenv("DATASET_PATH", str(DEFAULT_DATASET_PATH))

    def _resolve_dataset_path(self) -> Path:
        configured = Path(self.dataset_path).expanduser().resolve()
        if configured.exists():
            return configured

        data_dir = DEFAULT_DATASET_PATH.parent
        if configured.parent != data_dir.resolve():
            return configured

        candidates = sorted(
            [path for path in data_dir.glob("*.csv") if path.name != "sample.csv"],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0].resolve()
        return configured

    @staticmethod
    def _write_bootstrap_dataset(path: Path) -> None:
        """Create a tiny fallback dataset so API stays operational."""
        path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap = pd.DataFrame(
            [
                {
                    "date": "2024-01-10",
                    "sentiment": "4",
                    "text": "AI is changing developer workflows fast #AI #Productivity",
                },
                {
                    "date": "2024-01-11",
                    "sentiment": "0",
                    "text": "I am worried about misuse of AI content generation #AI #Safety",
                },
                {
                    "date": "2024-02-01",
                    "sentiment": "4",
                    "text": "Python and LangChain are great for quick agent prototypes #Python #AI",
                },
            ]
        )
        bootstrap.to_csv(path, index=False)

    @staticmethod
    def _has_kaggle_credentials() -> bool:
        return bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))

    def _ensure_dataset_exists(
        self,
        path: Path,
        force_refresh: bool = False,
        prefer_kaggle: bool = False,
        strict_kaggle: bool = False,
        max_rows_override: int | None = None,
        keyword_filter: str | None = None,
    ) -> None:
        if path.exists() and not force_refresh and not prefer_kaggle:
            return

        dataset_slug = os.getenv("KAGGLE_DATASET", "kazanova/sentiment140")
        max_rows = max_rows_override or int(os.getenv("KAGGLE_MAX_ROWS", "1000"))
        selected_file = os.getenv("KAGGLE_FILE", "").strip() or None
        had_existing = path.exists()
        can_attempt_kaggle = self._has_kaggle_credentials()

        if not can_attempt_kaggle:
            if strict_kaggle:
                raise ValueError(
                    "Kaggle fetch requested but credentials are missing. "
                    "Set KAGGLE_USERNAME and KAGGLE_KEY."
                )
            if had_existing:
                return
            self._write_bootstrap_dataset(path)
            return

        try:
            fetch_kaggle_dataset_to_csv(
                dataset=dataset_slug,
                output_csv=path,
                selected_file=selected_file,
                max_rows=max_rows,
                keyword_filter=keyword_filter,
            )
        except BaseException as exc:
            if strict_kaggle:
                raise ValueError(f"Kaggle fetch failed: {exc}") from exc
            if had_existing:
                # Keep existing local dataset when refresh attempt fails.
                return
            # Keep service usable even when Kaggle creds/network are unavailable.
            self._write_bootstrap_dataset(path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at {path}. Auto-fetch failed and fallback creation failed. "
                    "Set KAGGLE_USERNAME and KAGGLE_KEY (and install data deps) "
                    "or provide a local data/sample.csv."
                ) from exc

    def _load_playwright_rows(self, keyword: str, limit: int) -> pd.DataFrame:
        rows = fetch_public_x_with_playwright(keyword=keyword, limit=limit)
        if not rows:
            return pd.DataFrame(columns=["text", "sentiment", "date"])
        return self._normalize_columns(pd.DataFrame(rows))

    def _read_dataset(self, path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")
        if df.empty:
            raise ValueError("Dataset is empty")

        # Sentiment140 raw files are headerless; if first row becomes header,
        # re-read with explicit positional columns to retain all rows.
        lower_columns = [str(column).lower() for column in df.columns]
        if (
            len(lower_columns) >= 6
            and lower_columns[0] in {"0", "2", "4"}
            and "text" not in lower_columns
        ):
            return pd.read_csv(
                path,
                header=None,
                names=["sentiment", "id", "date", "query", "user", "text"],
                usecols=["sentiment", "date", "text"],
                low_memory=False,
            )
        return df

    def _load(self) -> pd.DataFrame:
        path = self._resolve_dataset_path()
        self._ensure_dataset_exists(path, prefer_kaggle=True)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
        return self._normalize_columns(self._read_dataset(path))

    def _load_with_kaggle_refresh(self, keyword: str) -> pd.DataFrame:
        path = self._resolve_dataset_path()
        runtime_max_rows = int(os.getenv("KAGGLE_MAX_ROWS_RUNTIME", "1000"))
        runtime_max_rows = max(200, min(runtime_max_rows, 5000))
        self._ensure_dataset_exists(
            path,
            force_refresh=True,
            strict_kaggle=True,
            max_rows_override=runtime_max_rows,
            keyword_filter=keyword,
        )
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
        return self._normalize_columns(self._read_dataset(path))

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        column_map: dict[str, str] = {}
        lower_to_original = {column.lower(): column for column in df.columns}

        text_candidates = ["text", "tweet", "content"]
        sentiment_candidates = ["sentiment", "target", "label"]
        date_candidates = ["date", "created_at", "timestamp"]

        for candidate in text_candidates:
            if candidate in lower_to_original:
                column_map[lower_to_original[candidate]] = "text"
                break

        for candidate in sentiment_candidates:
            if candidate in lower_to_original:
                column_map[lower_to_original[candidate]] = "sentiment"
                break

        for candidate in date_candidates:
            if candidate in lower_to_original:
                column_map[lower_to_original[candidate]] = "date"
                break

        normalized = df.rename(columns=column_map).copy()

        # Sentiment140 raw format fallback (no header):
        # sentiment,id,date,query,user,text
        if "text" not in normalized.columns and normalized.shape[1] >= 6:
            normalized = pd.DataFrame(
                {
                    "sentiment": normalized.iloc[:, 0],
                    "date": normalized.iloc[:, 2],
                    "text": normalized.iloc[:, 5],
                }
            )

        if "text" not in normalized.columns:
            raise ValueError("Dataset must include a text/tweet/content column")

        if "sentiment" not in normalized.columns:
            normalized["sentiment"] = "unknown"

        if "date" not in normalized.columns:
            normalized["date"] = pd.NaT

        normalized["text"] = normalized["text"].astype(str).map(clean_text)
        normalized["sentiment"] = normalized["sentiment"].map(map_sentiment)
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
        return normalized

    def run(
        self,
        keyword: str,
        limit: int = 50,
        sentiment_filter: str | None = None,
        since_date: str | None = None,
        source: str = "dataset",
    ) -> dict[str, Any]:
        """Filter and summarize posts by keyword and optional constraints."""
        if not keyword or not keyword.strip():
            raise ValueError("keyword is required")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        if source not in {"dataset", "playwright", "kaggle"}:
            raise ValueError("source must be 'dataset', 'kaggle', or 'playwright'")

        df = (
            self._load_playwright_rows(keyword=keyword, limit=limit)
            if source == "playwright"
            else self._load_with_kaggle_refresh(keyword=keyword) if source == "kaggle" else self._load()
        )
        keyword_value = keyword.strip().lower()
        escaped_keyword = re.escape(keyword_value)
        if re.fullmatch(r"\w+", keyword_value):
            keyword_pattern = rf"\b{escaped_keyword}\b"
        else:
            keyword_pattern = escaped_keyword

        filtered = df[
            df["text"].str.lower().str.contains(keyword_pattern, regex=True, na=False)
        ].copy()

        if since_date:
            since = pd.to_datetime(since_date, errors="coerce")
            if pd.isna(since):
                raise ValueError("since_date must be a valid date string")
            filtered = filtered[filtered["date"] >= since]

        if sentiment_filter:
            sentiment = map_sentiment(sentiment_filter)
            filtered = filtered[filtered["sentiment"] == sentiment]

        filtered = filtered.sort_values(by="date", ascending=False, na_position="last")
        limited = filtered.head(limit)

        sentiment_distribution = (
            limited["sentiment"].value_counts(dropna=False).to_dict() if not limited.empty else {}
        )

        posts = [
            {
                "text": row["text"],
                "sentiment": row["sentiment"],
                "date": row["date"].isoformat() if pd.notna(row["date"]) else None,
            }
            for _, row in limited.iterrows()
        ]

        return {
            "keyword": keyword,
            "source": source,
            "total_matches": int(filtered.shape[0]),
            "returned_count": int(limited.shape[0]),
            "sentiment_distribution": sentiment_distribution,
            "top_hashtags": extract_hashtags(limited["text"].tolist()),
            "posts": posts,
        }
