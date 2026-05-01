"""Kaggle dataset helpers for on-demand dataset refresh."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def _normalize_to_app_schema(input_path: Path, output_path: Path, max_rows: int) -> None:
    """Normalize arbitrary Kaggle CSV into columns: date,sentiment,text."""
    encoding_attempts = ["utf-8", "latin-1", "cp1252"]
    dataframe: pd.DataFrame | None = None
    last_error: Exception | None = None

    for encoding in encoding_attempts:
        try:
            dataframe = pd.read_csv(input_path, nrows=max_rows, encoding=encoding, low_memory=False)
            break
        except Exception as exc:  # pragma: no cover - depends on source CSV
            last_error = exc

    if dataframe is None:
        raise ValueError(f"Unable to read CSV {input_path} with supported encodings.") from last_error

    normalized: pd.DataFrame | None = None
    lowered = {str(column).lower(): column for column in dataframe.columns}

    text_column = None
    sentiment_column = None
    date_column = None

    for candidate in ("text", "tweet", "content"):
        if candidate in lowered:
            text_column = lowered[candidate]
            break
    for candidate in ("sentiment", "target", "label"):
        if candidate in lowered:
            sentiment_column = lowered[candidate]
            break
    for candidate in ("date", "created_at", "timestamp"):
        if candidate in lowered:
            date_column = lowered[candidate]
            break

    if text_column is not None:
        normalized = pd.DataFrame(
            {
                "date": dataframe[date_column] if date_column is not None else pd.NA,
                "sentiment": dataframe[sentiment_column] if sentiment_column is not None else "unknown",
                "text": dataframe[text_column],
            }
        )

    if normalized is None and dataframe.shape[1] >= 6:
        sentiment140 = pd.read_csv(
            input_path,
            header=None,
            names=["sentiment", "id", "date", "query", "user", "text"],
            usecols=["sentiment", "date", "text"],
            nrows=max_rows,
            encoding="latin-1",
            low_memory=False,
        )
        normalized = sentiment140[["date", "sentiment", "text"]]

    if normalized is None:
        raise ValueError(
            "Could not normalize dataset to required columns. Expected either named text columns or "
            "Sentiment140-like 6-column format."
        )

    normalized["text"] = normalized["text"].astype(str)
    normalized["sentiment"] = normalized["sentiment"].astype(str)
    normalized["date"] = normalized["date"].astype(str)
    normalized = normalized.dropna(subset=["text"])
    normalized = normalized[normalized["text"].str.strip() != ""]
    normalized.to_csv(output_path, index=False)


def fetch_kaggle_dataset_to_csv(
    *,
    dataset: str,
    output_csv: Path,
    selected_file: str | None = None,
    max_rows: int = 100000,
) -> Path:
    """Download and normalize a Kaggle dataset to an app-compatible CSV."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise ValueError(
            "Kaggle package is not installed. Install with `pip install -e .[data]`."
        ) from exc

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    files = api.dataset_list_files(dataset).files
    file_names = [file.name for file in files]

    selected = selected_file
    if selected is None:
        csv_files = [name for name in file_names if name.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError(f"No CSV files found in dataset '{dataset}'. Available files: {file_names}")
        selected = csv_files[0]

    if selected not in file_names:
        raise ValueError(
            f"File '{selected}' not found in dataset '{dataset}'. Available files: {file_names}"
        )

    output_dir = output_csv.parent
    api.dataset_download_file(dataset=dataset, file_name=selected, path=str(output_dir), force=True)

    downloaded = output_dir / selected
    zipped_download = output_dir / f"{selected}.zip"
    if zipped_download.exists() and not downloaded.exists():
        with ZipFile(zipped_download, "r") as archive:
            archive.extractall(output_dir)

    if not downloaded.exists():
        candidates = list(output_dir.rglob(selected))
        if not candidates:
            raise FileNotFoundError(f"Downloaded file '{selected}' was not found under {output_dir}.")
        downloaded = candidates[0]

    _normalize_to_app_schema(downloaded, output_csv, max_rows=max_rows)
    return output_csv
