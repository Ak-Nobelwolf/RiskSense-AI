import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def fetch_earthquakes(lat: float, lon: float, max_radius_km: int = 300, min_magnitude: float = 2.5) -> list[dict]:
    params = {
        "format": "geojson",
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": max_radius_km,
        "minmagnitude": min_magnitude,
        "orderby": "time",
        "limit": 50,
    }
    try:
        resp = httpx.get(USGS_URL, params=params, timeout=30)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        results = []
        for f in features:
            props = f.get("properties", {})
            coords = f.get("geometry", {}).get("coordinates", [0, 0, 0])
            ts_ms = props.get("time", 0)
            results.append({
                "id": f.get("id", ""),
                "title": props.get("title", "Earthquake"),
                "magnitude": float(props.get("mag", 0)),
                "depth_km": float(coords[2]) if len(coords) > 2 else 0,
                "latitude": float(coords[1]),
                "longitude": float(coords[0]),
                "timestamp": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                "status": "open",
                "category": "Earthquakes",
            })
        logger.info("Fetched %d earthquakes near (%.2f, %.2f)", len(results), lat, lon)
        return results
    except Exception as e:
        logger.warning("USGS request failed: %s", e)
        return []


def generate_mock_earthquakes(lat: float, lon: float) -> list[dict]:
    np.random.seed(44)
    n = np.random.poisson(3)
    results = []
    now = datetime.now(timezone.utc)
    for i in range(n):
        offset_lat = lat + np.random.uniform(-3, 3)
        offset_lon = lon + np.random.uniform(-3, 3)
        dist = np.sqrt((offset_lat - lat)**2 + (offset_lon - lon)**2) * 111
        if dist > 300:
            continue
        mag = round(np.random.uniform(2.5, 5.5), 1)
        results.append({
            "id": f"mock_eq_{i}",
            "title": f"M{mag} Earthquake",
            "magnitude": mag,
            "depth_km": round(np.random.uniform(2, 30), 1),
            "latitude": round(offset_lat, 4),
            "longitude": round(offset_lon, 4),
            "timestamp": now - timedelta(hours=np.random.randint(0, 72)),
            "status": "open",
            "category": "Earthquakes",
        })
    return results


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")


def run(lat: float | None = None, lon: float | None = None, location_name: str = "mumbai", force_refresh: bool = False, max_radius_km: int = 300) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slug(location_name)

    cached = sorted(RAW_DIR.glob(f"earthquakes__{slug}.parquet"))
    if cached and not force_refresh:
        df = pd.read_parquet(cached[-1])
        logger.info("Reusing cached earthquakes for '%s': %s (%d rows)", slug, cached[-1].name, len(df))
        return df

    if lat and lon:
        events = fetch_earthquakes(lat, lon, max_radius_km)
    else:
        events = []
    if not events:
        logger.info("No USGS earthquakes — falling back to mock")
        events = generate_mock_earthquakes(lat or 19.076, lon or 72.8777)

    df = pd.DataFrame(events)
    if df.empty:
        logger.warning("No earthquake data for '%s'", slug)
        return df

    from src.ingestion.storage_writer import store_parquet
    path = store_parquet(df, "earthquakes.parquet", subdir="raw", location_slug=slug)
    logger.info("Saved %d earthquake rows to %s for '%s'", len(df), path, slug)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
