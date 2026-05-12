"""Download both source datasets from Kaggle into data/raw/.

Reads KAGGLE_USERNAME and KAGGLE_KEY from .env and writes a temporary
~/.kaggle/kaggle.json the Kaggle CLI expects. After download, files are
renamed to the canonical names referenced throughout the project.

Usage:
    python scripts/download_data.py             # both datasets
    python scripts/download_data.py --paysim    # only PaySim
    python scripts/download_data.py --custom    # only custom
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

from src.config import (
    CUSTOM_CSV,
    CUSTOM_KAGGLE_SLUG,
    KAGGLE_KEY,
    KAGGLE_USERNAME,
    PAYSIM_CSV,
    PAYSIM_KAGGLE_SLUG,
    RAW_DIR,
)


def _write_kaggle_json() -> Path:
    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        raise RuntimeError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in .env "
            "(get them at https://www.kaggle.com/settings)."
        )
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    creds_file = kaggle_dir / "kaggle.json"
    creds_file.write_text(json.dumps({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}))
    os.chmod(creds_file, 0o600)
    return creds_file


def _download_slug(slug: str, dest_name: Path) -> None:
    """Download a Kaggle dataset slug, unzip, and rename the largest CSV
    inside to dest_name."""
    _write_kaggle_json()
    # Import lazily — kaggle inspects ~/.kaggle/kaggle.json at import time
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    tmp_dir = RAW_DIR / f"_tmp_{slug.replace('/', '_')}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] downloading {slug} -> {tmp_dir}")
    api.dataset_download_files(slug, path=str(tmp_dir), quiet=False, unzip=False)

    # Unzip everything
    for zp in tmp_dir.glob("*.zip"):
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(tmp_dir)
        zp.unlink()

    # Find the largest CSV (usually the main file)
    csvs = sorted(tmp_dir.glob("**/*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csvs:
        raise RuntimeError(f"No CSVs found in download for {slug}")

    dest_name.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(csvs[0]), dest_name)
    # Tidy up
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[kaggle] saved -> {dest_name} ({dest_name.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paysim", action="store_true", help="download only PaySim")
    parser.add_argument("--custom", action="store_true", help="download only custom dataset")
    args = parser.parse_args()

    do_paysim = args.paysim or not args.custom
    do_custom = args.custom or not args.paysim

    if do_paysim:
        if PAYSIM_CSV.exists():
            print(f"[skip] {PAYSIM_CSV.name} already present")
        else:
            _download_slug(PAYSIM_KAGGLE_SLUG, PAYSIM_CSV)

    if do_custom:
        if CUSTOM_CSV.exists():
            print(f"[skip] {CUSTOM_CSV.name} already present")
        else:
            _download_slug(CUSTOM_KAGGLE_SLUG, CUSTOM_CSV)


if __name__ == "__main__":
    main()
