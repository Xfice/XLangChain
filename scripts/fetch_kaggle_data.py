"""Download a Kaggle dataset CSV into the local data folder."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a Kaggle dataset file and store it as local CSV input."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Kaggle dataset slug, e.g. kazanova/sentiment140",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Specific file name from dataset (optional). Defaults to first CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to place the downloaded file.",
    )
    parser.add_argument(
        "--target-name",
        default="sample.csv",
        help="Final local file name (used by the app), default: sample.csv",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100000,
        help="Maximum rows to keep in output CSV to control memory/size (default: 100000).",
    )
    return parser.parse_args()


def _normalize_to_app_schema(input_path: Path, output_path: Path, max_rows: int) -> None:
    """Normalize arbitrary Kaggle CSV into columns: date,sentiment,text."""
    lower_encoding_attempts = ["utf-8", "latin-1", "cp1252"]
    dataframe: pd.DataFrame | None = None
    last_error: Exception | None = None

    # First try: header-based read for normal CSV datasets.
    for encoding in lower_encoding_attempts:
        try:
            dataframe = pd.read_csv(input_path, nrows=max_rows, encoding=encoding, low_memory=False)
            break
        except Exception as exc:
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

    # Sentiment140 fallback: 6 columns with no header.
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    files = api.dataset_list_files(args.dataset).files
    file_names = [file.name for file in files]
    selected = args.file

    if selected is None:
        csv_files = [name for name in file_names if name.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError(
                f"No CSV files found in dataset '{args.dataset}'. Available files: {file_names}"
            )
        selected = csv_files[0]

    if selected not in file_names:
        raise ValueError(
            f"File '{selected}' not found in dataset '{args.dataset}'. Available files: {file_names}"
        )

    api.dataset_download_file(
        dataset=args.dataset,
        file_name=selected,
        path=str(output_dir),
        force=True,
    )

    downloaded = output_dir / selected
    zipped_download = output_dir / f"{selected}.zip"
    if zipped_download.exists() and not downloaded.exists():
        with ZipFile(zipped_download, "r") as archive:
            archive.extractall(output_dir)

    if not downloaded.exists():
        # Kaggle may create nested paths for some datasets; fallback search.
        candidates = list(output_dir.rglob(selected))
        if not candidates:
            raise FileNotFoundError(
                f"Downloaded file '{selected}' was not found under {output_dir}."
            )
        downloaded = candidates[0]

    target = output_dir / args.target_name
    _normalize_to_app_schema(downloaded, target, max_rows=args.max_rows)

    print(f"Downloaded '{selected}' from '{args.dataset}' and normalized to '{target}'.")


if __name__ == "__main__":
    main()
