"""Reusable LangChain-compatible tool for public X data analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.processing import clean_text, extract_hashtags, map_sentiment
from app.sources.playwright_source import fetch_public_x_with_playwright


@dataclass
class TwitterDataTool:
    """Analyze public X/Twitter-like posts from a local CSV dataset."""

    dataset_path: str | Path = "data/sample.csv"

    def _load_playwright_rows(self, keyword: str, limit: int) -> pd.DataFrame:
        rows = fetch_public_x_with_playwright(keyword=keyword, limit=limit)
        if not rows:
            return pd.DataFrame(columns=["text", "sentiment", "date"])
        return self._normalize_columns(pd.DataFrame(rows))

    def _load(self) -> pd.DataFrame:
        path = Path(self.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("Dataset is empty")
        return self._normalize_columns(df)

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

        if source not in {"dataset", "playwright"}:
            raise ValueError("source must be 'dataset' or 'playwright'")

        df = (
            self._load_playwright_rows(keyword=keyword, limit=limit)
            if source == "playwright"
            else self._load()
        )
        keyword_value = keyword.strip().lower()
        filtered = df[df["text"].str.lower().str.contains(keyword_value, na=False)].copy()

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
