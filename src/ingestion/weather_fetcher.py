import math
import os
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OWM_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
WAPI_KEY = os.getenv("WEATHERAPI_KEY", "")
DEFAULT_LAT = float(os.getenv("TARGET_LATITUDE", "19.0760"))
DEFAULT_LON = float(os.getenv("TARGET_LONGITUDE", "72.8777"))
FORECAST_DAYS = int(os.getenv("WEATHER_FORECAST_DAYS", "2"))
RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))

OWM_BASE = "https://api.openweathermap.org/data/2.5"


def _solar_elevation(lat: float, lon: float, dt: datetime) -> float:
    n = dt.timetuple().tm_yday
    decl = 23.44 * math.sin(math.radians(284 + n) * 360 / 365)
    solar_time = dt.hour + dt.minute / 60 + lon / 15
    hour_angle = 15 * (solar_time - 12)
    lat_r, decl_r, ha_r = math.radians(lat), math.radians(decl), math.radians(hour_angle)
    sin_alt = math.sin(lat_r) * math.sin(decl_r) + math.cos(lat_r) * math.cos(decl_r) * math.cos(ha_r)
    return math.degrees(math.asin(max(-1, min(1, sin_alt))))


def _estimate_uvi(lat: float, lon: float, dt: datetime) -> float:
    elev = _solar_elevation(lat, lon, dt)
    if elev <= 0:
        return 0.0
    lat_abs = abs(lat)
    peak_uvi = 12.0 if lat_abs < 15 else (10.0 if lat_abs < 30 else (7.0 if lat_abs < 45 else 4.0))
    return round(max(0, peak_uvi * (math.sin(math.radians(elev)) ** 1.3)), 1)


def fetch_owm_forecast(lat: float, lon: float) -> dict | None:
    if not OWM_API_KEY:
        logger.warning("OPENWEATHERMAP_API_KEY not set")
        return None
    url = f"{OWM_BASE}/forecast"
    params = {"lat": lat, "lon": lon, "units": "metric", "appid": OWM_API_KEY}
    try:
        resp = httpx.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("OWM forecast request failed: %s", e)
        return None


def fetch_owm_current(lat: float, lon: float) -> dict | None:
    if not OWM_API_KEY:
        return None
    url = f"{OWM_BASE}/weather"
    params = {"lat": lat, "lon": lon, "units": "metric", "appid": OWM_API_KEY}
    try:
        resp = httpx.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("OWM current weather request failed: %s", e)
        return None


def _get_precip(entry: dict, key: str = "rain") -> float:
    if key in entry and isinstance(entry[key], dict):
        return float(entry[key].get("3h", 0))
    return 0.0


def normalize_owm_forecast(data: dict, lat: float, lon: float) -> pd.DataFrame:
    rows = []
    for entry in data.get("list", []):
        ts = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
        rows.append({
            "forecast_timestamp": ts,
            "temperature": entry["main"]["temp"],
            "humidity": entry["main"]["humidity"],
            "wind_speed": entry["wind"]["speed"],
            "wind_gust": float(entry["wind"].get("gust", 0)),
            "wind_direction": float(entry["wind"].get("deg", 0)),
            "precipitation": _get_precip(entry, "rain") + _get_precip(entry, "snow"),
            "pressure": entry["main"]["pressure"],
            "uv_index": _estimate_uvi(lat, lon, ts),
            "condition_text": entry["weather"][0]["description"].title() if entry.get("weather") else "Clear",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("forecast_timestamp").set_index("forecast_timestamp")
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    df_num = df[num_cols].resample("1h").interpolate(method="time")
    df_cat = df[cat_cols].resample("1h").ffill()
    df = pd.concat([df_num, df_cat], axis=1).reset_index()
    return df


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")


def generate_mock_forecast(lat: float, lon: float, days: int = 2) -> pd.DataFrame:
    np.random.seed(42)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    rows = []
    for d in range(days):
        for h in range(24):
            ts = now + timedelta(days=d, hours=h)
            hour_of_day = ts.hour
            base_temp = 28 + 6 * np.sin(np.pi * (hour_of_day - 8) / 12)
            rows.append({
                "forecast_timestamp": ts.isoformat(),
                "temperature": round(base_temp + np.random.normal(0, 1.5), 1),
                "humidity": round(65 + 15 * np.sin(np.pi * (hour_of_day - 14) / 12) + np.random.normal(0, 5), 0),
                "wind_speed": round(12 + 8 * np.sin(np.pi * hour_of_day / 12) + np.random.normal(0, 3), 1),
                "wind_gust": round((12 + 8 * np.sin(np.pi * hour_of_day / 12)) * 1.4 + np.random.normal(0, 4), 1),
                "wind_direction": round(np.random.uniform(0, 360), 0),
                "precipitation": round(np.random.exponential(0.5 if 8 <= hour_of_day <= 20 else 0.2), 2),
                "pressure": round(1008 + np.random.normal(0, 3), 1),
                "uv_index": round(max(0, 8 * np.sin(np.pi * (hour_of_day - 6) / 12) + np.random.normal(0, 0.5)), 1),
                "condition_text": "Sunny" if 8 <= hour_of_day <= 18 else "Clear",
            })
    df = pd.DataFrame(rows)
    df["forecast_timestamp"] = pd.to_datetime(df["forecast_timestamp"])
    logger.info("Generated %d rows of mock weather data for lat=%s,lon=%s", len(df), lat, lon)
    return df


def run(lat: float | None = None, lon: float | None = None, location_name: str = "mumbai", force_refresh: bool = False) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    lat = lat if lat is not None else DEFAULT_LAT
    lon = lon if lon is not None else DEFAULT_LON
    slug = _slug(location_name)

    cached = sorted(RAW_DIR.glob(f"weather__{slug}.parquet"))
    if cached and not force_refresh:
        df = pd.read_parquet(cached[-1])
        logger.info("Reusing cached weather data for '%s': %s (%d rows)", slug, cached[-1].name, len(df))
        return df

    df = pd.DataFrame()
    data = fetch_owm_forecast(lat, lon)
    if data:
        df = normalize_owm_forecast(data, lat, lon)
        current = fetch_owm_current(lat, lon)
        if current and df.empty:
            ts = datetime.now(timezone.utc)
            df = pd.DataFrame([{
                "forecast_timestamp": ts,
                "temperature": current["main"]["temp"],
                "humidity": current["main"]["humidity"],
                "wind_speed": current["wind"]["speed"],
                "wind_gust": float(current["wind"].get("gust", 0)),
                "wind_direction": float(current["wind"].get("deg", 0)),
                "precipitation": 0.0,
                "pressure": current["main"]["pressure"],
                "uv_index": _estimate_uvi(lat, lon, ts),
                "condition_text": current["weather"][0]["description"].title() if current.get("weather") else "Clear",
            }])
    else:
        logger.info("No OWM data — falling back to mock forecast")
        df = generate_mock_forecast(lat, lon, FORECAST_DAYS)

    if df.empty:
        logger.warning("No weather data retrieved for '%s'", slug)
        return df

    from src.ingestion.storage_writer import store_parquet
    path = store_parquet(df, "weather.parquet", subdir="raw", location_slug=slug)
    logger.info("Saved %d weather rows to %s for '%s'", len(df), path, slug)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
