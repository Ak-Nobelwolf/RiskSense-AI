import os
import logging
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OWM_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
OWM_BASE = "https://api.openweathermap.org/data/2.5"


def _aqi_label(aqi: int) -> str:
    return {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}.get(aqi, "Unknown")


def fetch_air_pollution(lat: float, lon: float) -> list[dict] | None:
    if not OWM_API_KEY:
        logger.warning("OPENWEATHERMAP_API_KEY not set — cannot fetch air quality")
        return None
    url = f"{OWM_BASE}/air_pollution/forecast"
    params = {"lat": lat, "lon": lon, "appid": OWM_API_KEY}
    try:
        resp = httpx.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("list", [])
    except Exception as e:
        logger.warning("Air pollution request failed: %s", e)
        return None


def generate_mock_pollution(lat: float, lon: float, hours: int = 48) -> pd.DataFrame:
    np.random.seed(43)
    now = pd.Timestamp.now(tz="UTC").floor("h")
    rows = []
    for i in range(hours):
        ts = now + pd.Timedelta(hours=i)
        base_aqi = np.random.choice([1, 1, 1, 1, 2, 2, 2, 3, 3, 4], p=[0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01])
        rows.append({
            "timestamp": ts,
            "aqi": int(base_aqi),
            "aqi_label": _aqi_label(base_aqi),
            "co": round(np.random.uniform(100, 500), 1),
            "no": round(np.random.uniform(0, 2), 2),
            "no2": round(np.random.uniform(0.5, 5), 1),
            "o3": round(np.random.uniform(10, 80), 1),
            "so2": round(np.random.uniform(0.1, 2), 2),
            "pm2_5": round(np.random.uniform(0.3, 8), 1),
            "pm10": round(np.random.uniform(0.5, 15), 1),
            "nh3": round(np.random.uniform(0.1, 3), 2),
        })
    df = pd.DataFrame(rows)
    logger.info("Generated %d rows of mock air quality data", len(df))
    return df


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")


def run(lat: float | None = None, lon: float | None = None, location_name: str = "mumbai", force_refresh: bool = False) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slug(location_name)

    cached = sorted(RAW_DIR.glob(f"air_quality__{slug}.parquet"))
    if cached and not force_refresh:
        df = pd.read_parquet(cached[-1])
        logger.info("Reusing cached air quality for '%s': %s (%d rows)", slug, cached[-1].name, len(df))
        return df

    data = fetch_air_pollution(lat, lon) if lat and lon else None
    if data:
        rows = []
        for entry in data:
            ts = pd.Timestamp(entry["dt"], unit="s", tz="UTC")
            aqi = entry["main"]["aqi"]
            comp = entry["components"]
            rows.append({
                "timestamp": ts,
                "aqi": aqi,
                "aqi_label": _aqi_label(aqi),
                "co": comp.get("co", 0),
                "no": comp.get("no", 0),
                "no2": comp.get("no2", 0),
                "o3": comp.get("o3", 0),
                "so2": comp.get("so2", 0),
                "pm2_5": comp.get("pm2_5", 0),
                "pm10": comp.get("pm10", 0),
                "nh3": comp.get("nh3", 0),
            })
        df = pd.DataFrame(rows)
    else:
        logger.info("No air pollution data — falling back to mock")
        df = generate_mock_pollution(lat or 19.076, lon or 72.8777)

    if df.empty:
        logger.warning("No air quality data for '%s'", slug)
        return df

    from src.ingestion.storage_writer import store_parquet
    path = store_parquet(df, "air_quality.parquet", subdir="raw", location_slug=slug)
    logger.info("Saved %d air quality rows to %s for '%s'", len(df), path, slug)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
