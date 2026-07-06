import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.ingestion.gcs_client import (
    is_available as gcs_available,
    upload_parquet as gcs_upload,
    download_parquet as gcs_download,
    list_blobs as gcs_list,
    raws_prefix,
    processed_prefix,
)

load_dotenv()

logger = logging.getLogger(__name__)

RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))


def _use_gcs() -> bool:
    return gcs_available()


def _gcs_blob_name(filename: str, subdir: str, location_slug: str | None = None) -> str:
    if location_slug:
        stem, ext = os.path.splitext(filename)
        filename = f"{stem}__{location_slug}{ext}"
    prefix = raws_prefix() if subdir == "raw" else processed_prefix()
    return f"{prefix}{filename}"


def store_parquet(df: pd.DataFrame, filename: str, subdir: str = "raw", location_slug: str | None = None) -> Path:
    if _use_gcs():
        blob_name = _gcs_blob_name(filename, subdir, location_slug)
        gcs_upload(df, blob_name)
    if location_slug:
        stem, ext = os.path.splitext(filename)
        filename = f"{stem}__{location_slug}{ext}"
    out = (RAW_DIR if subdir == "raw" else PROCESSED_DIR) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Stored %d rows to %s", len(df), out)
    return out


def load_latest_parquet(pattern: str, subdir: str = "raw", location_slug: str | None = None) -> pd.DataFrame | None:
    if _use_gcs():
        blob_name = _gcs_blob_name(pattern, subdir, location_slug)
        df = gcs_download(blob_name)
        if df is not None:
            return df
    base = RAW_DIR if subdir == "raw" else PROCESSED_DIR
    if not base.exists():
        return None
    if location_slug:
        stem, ext = os.path.splitext(pattern)
        pattern = f"{stem}__{location_slug}{ext}"
    files = sorted(base.glob(pattern))
    if not files:
        return None
    latest = files[-1]
    df = pd.read_parquet(latest)
    logger.info("Loaded %d rows from %s", len(df), latest)
    return df


def list_raw_files() -> list[Path | str]:
    if _use_gcs():
        return gcs_list(prefix=raws_prefix())
    if not RAW_DIR.exists():
        return []
    return sorted(RAW_DIR.glob("*.parquet"))
