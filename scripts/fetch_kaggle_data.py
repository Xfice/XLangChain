"""Download a Kaggle dataset CSV into the local data folder."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

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
    return parser.parse_args()


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
    target.write_bytes(downloaded.read_bytes())
    if downloaded != target:
        downloaded.unlink(missing_ok=True)

    print(f"Downloaded '{selected}' from '{args.dataset}' to '{target}'.")


if __name__ == "__main__":
    main()
