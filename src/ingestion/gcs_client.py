from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GCS_AVAILABLE = bool(os.getenv("GCP_PROJECT_ID")) and (
    bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")) or bool(os.getenv("K_SERVICE"))
)

_client = None
_bucket = None


def _get_client():
    global _client
    if _client is None and GCS_AVAILABLE:
        from google.cloud import storage
        _client = storage.Client()
    return _client


def _get_bucket():
    global _bucket
    if _bucket is None and GCS_AVAILABLE:
        client = _get_client()
        if client:
            bucket_name = os.getenv("GCS_BUCKET_NAME", "")
            if bucket_name:
                _bucket = client.bucket(bucket_name)
    return _bucket


def is_available() -> bool:
    return GCS_AVAILABLE and _get_bucket() is not None


def upload_parquet(df: pd.DataFrame, blob_name: str) -> str | None:
    if not is_available():
        return None
    try:
        bucket = _get_bucket()
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        blob = bucket.blob(blob_name)
        blob.upload_from_file(buf, content_type="application/octet-stream")
        logger.info("Uploaded %d rows to gs://%s/%s", len(df), bucket.name, blob_name)
        return blob_name
    except Exception as e:
        logger.warning("GCS upload failed for %s: %s", blob_name, e)
        return None


def download_parquet(blob_name: str) -> pd.DataFrame | None:
    if not is_available():
        return None
    try:
        bucket = _get_bucket()
        blob = bucket.blob(blob_name)
        if not blob.exists():
            return None
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        df = pd.read_parquet(buf)
        logger.info("Downloaded %d rows from gs://%s/%s", len(df), bucket.name, blob_name)
        return df
    except Exception as e:
        logger.warning("GCS download failed for %s: %s", blob_name, e)
        return None


def list_blobs(prefix: str = "") -> list[str]:
    if not is_available():
        return []
    try:
        bucket = _get_bucket()
        blobs = list(bucket.list_blobs(prefix=prefix))
        return [b.name for b in blobs]
    except Exception as e:
        logger.warning("GCS list blobs failed for prefix %s: %s", prefix, e)
        return []


def blob_exists(blob_name: str) -> bool:
    if not is_available():
        return False
    try:
        bucket = _get_bucket()
        return bucket.blob(blob_name).exists()
    except Exception:
        return False


def delete_blob(blob_name: str) -> bool:
    if not is_available():
        return False
    try:
        bucket = _get_bucket()
        bucket.blob(blob_name).delete()
        return True
    except Exception as e:
        logger.warning("GCS delete failed for %s: %s", blob_name, e)
        return False


def raws_prefix() -> str:
    return "raw/"


def processed_prefix() -> str:
    return "processed/"
