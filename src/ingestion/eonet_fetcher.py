import os
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

EONET_BASE_URL = os.getenv("EONET_BASE_URL", "https://eonet.gsfc.nasa.gov/api/v3")
EONET_STATUS = os.getenv("EONET_STATUS", "open")
EONET_LIMIT = int(os.getenv("EONET_LIMIT", "100"))
RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))


def fetch_events(status: str = EONET_STATUS, limit: int = EONET_LIMIT) -> list[dict]:
    url = f"{EONET_BASE_URL}/events"
    params = {"status": status, "limit": limit}
    logger.info("Fetching EONET events: %s", url)
    response = httpx.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    events = data.get("events", [])
    logger.info("Fetched %d EONET events", len(events))
    return events


def normalize_event(event: dict) -> dict | None:
    event_id = event.get("id")
    title = event.get("title", "")
    categories = event.get("categories", [])
    category = categories[0].get("title", "Unknown") if categories else "Unknown"
    geometry = event.get("geometry", [])
    if not geometry:
        return None
    latest = geometry[-1]
    event_timestamp = latest.get("date", "")
    coords = latest.get("coordinates", [])
    if len(coords) != 2:
        return None
    longitude, latitude = coords
    return {
        "event_id": event_id,
        "category": category,
        "title": title,
        "status": "open",
        "event_timestamp": event_timestamp,
        "latitude": latitude,
        "longitude": longitude,
    }


def run() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        events = fetch_events()
        rows = [normalize_event(e) for e in events]
        rows = [r for r in rows if r is not None]
        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("No EONET events retrieved — trying cache")
            return _load_cached()
        path = RAW_DIR / f"eonet_events_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved %d EONET events to %s", len(df), path)
        return df
    except Exception as e:
        logger.warning("EONET fetch failed: %s — falling back to cache", e)
        return _load_cached()


def _load_cached() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("eonet_events_*.parquet"))
    if files:
        df = pd.read_parquet(files[-1])
        logger.info("Loaded %d EONET events from cache: %s", len(df), files[-1].name)
        return df
    logger.warning("No cached EONET parquet found")
    return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
