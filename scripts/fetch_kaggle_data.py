"""Download a Kaggle dataset CSV into the local data folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.sources.kaggle_source import fetch_kaggle_dataset_to_csv


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
    parser.add_argument("--max-rows", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / args.target_name
    fetch_kaggle_dataset_to_csv(
        dataset=args.dataset,
        output_csv=target,
        selected_file=args.file,
        max_rows=args.max_rows,
    )
    print(f"Downloaded dataset '{args.dataset}' and normalized to '{target}'.")


if __name__ == "__main__":
    main()
