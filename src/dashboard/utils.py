from pathlib import Path

import pandas as pd
import yaml

from src import database_bq
from src.ingestion.storage_writer import load_latest_parquet

CONFIG_PATH = Path("configs/settings.yaml")
PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")


def _loc_slug(location: str | None = None) -> str | None:
    if location is None:
        return None
    return _slug(location)


def load_risk_data(location: str | None = None) -> pd.DataFrame:
    if database_bq.is_available():
        return database_bq.query("SELECT * FROM fact_environment_risk ORDER BY timestamp DESC")
    df = load_latest_parquet("scored_risk.parquet", subdir="processed", location_slug=_loc_slug(location))
    return df if df is not None else pd.DataFrame()


def load_feature_data(location: str | None = None) -> pd.DataFrame:
    if database_bq.is_available():
        return database_bq.query("SELECT * FROM fact_environment_risk ORDER BY timestamp DESC")
    df = load_latest_parquet("features.parquet", subdir="processed", location_slug=_loc_slug(location))
    return df if df is not None else pd.DataFrame()


def load_explanations(location: str | None = None) -> pd.DataFrame:
    if database_bq.is_available():
        return database_bq.query("SELECT * FROM raw_explanations ORDER BY timestamp DESC")
    df = load_latest_parquet("explanations.parquet", subdir="processed", location_slug=_loc_slug(location))
    return df if df is not None else pd.DataFrame()


def load_eonet_raw() -> pd.DataFrame:
    if database_bq.is_available():
        return database_bq.query("SELECT * FROM raw_eonet_events")
    files = sorted(RAW_DIR.glob("eonet_events_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])


def load_earthquake_raw(location: str | None = None) -> pd.DataFrame:
    if database_bq.is_available():
        return database_bq.query("SELECT * FROM raw_earthquakes ORDER BY timestamp DESC")
    if location is None:
        files = sorted(RAW_DIR.glob("earthquakes_*.parquet"))
    else:
        files = sorted(RAW_DIR.glob(f"earthquakes__{_slug(location)}.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])


def load_air_quality_raw(location: str | None = None) -> pd.DataFrame:
    if database_bq.is_available():
        return database_bq.query("SELECT * FROM raw_air_quality ORDER BY timestamp DESC")
    if location is None:
        files = sorted(RAW_DIR.glob("air_quality_*.parquet"))
    else:
        files = sorted(RAW_DIR.glob(f"air_quality__{_slug(location)}.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])


def load_weather_raw(location: str | None = None) -> pd.DataFrame:
    if database_bq.is_available():
        return database_bq.query("SELECT * FROM raw_weather ORDER BY forecast_timestamp DESC")
    if location is None:
        files = sorted(RAW_DIR.glob("weather_*.parquet"))
    else:
        files = sorted(RAW_DIR.glob(f"weather__{_slug(location)}.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])


def get_current_risk(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    df = df.sort_values("timestamp", ascending=False)
    return df.iloc[0]


def risk_color(score: float) -> str:
    if score <= 39:
        return "green"
    elif score <= 69:
        return "orange"
    return "red"


def risk_emoji(band: str) -> str:
    mapping = {
        "Safe": "Safe",
        "Caution": "Caution",
        "Avoid": "Avoid",
    }
    return mapping.get(band, "?")
