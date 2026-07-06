import logging

import pandas as pd

from src.transform.validate import validate_coordinates, validate_timestamps

logger = logging.getLogger(__name__)


def deduplicate(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="last")
    dropped = before - len(df)
    if dropped:
        logger.info("Deduplicated %d rows (keys: %s)", dropped, key_cols)
    return df


def normalize_timestamps(df: pd.DataFrame, ts_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in ts_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def handle_nulls(df: pd.DataFrame, fill_map: dict[str, float] | None = None) -> pd.DataFrame:
    if fill_map is None:
        fill_map = {
            "temperature": 25.0,
            "humidity": 50.0,
            "wind_speed": 0.0,
            "wind_direction": 0.0,
            "precipitation": 0.0,
            "pressure": 1013.0,
            "uv_index": 0.0,
        }
    for col, default in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
    return df


def clean_eonet_events(df: pd.DataFrame) -> pd.DataFrame:
    required = ["event_id", "category", "title", "latitude", "longitude", "event_timestamp"]
    if not all(c in df.columns for c in required):
        missing = [c for c in required if c not in df.columns]
        logger.error("EONET events missing columns: %s", missing)
        return df
    df = deduplicate(df, key_cols=["event_id"])
    df = normalize_timestamps(df, ["event_timestamp", "ingestion_ts"])
    df = validate_coordinates(df)
    return df


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    required = ["forecast_timestamp", "temperature", "humidity", "wind_speed"]
    if not all(c in df.columns for c in required):
        missing = [c for c in required if c not in df.columns]
        logger.error("Weather data missing columns: %s", missing)
        return df
    df = deduplicate(df, key_cols=["forecast_timestamp"])
    df = normalize_timestamps(df, ["forecast_timestamp"])
    df = handle_nulls(df)
    df = validate_timestamps(df, "forecast_timestamp")
    return df


def run_eonet(df: pd.DataFrame) -> pd.DataFrame:
    return clean_eonet_events(df)


def run_weather(df: pd.DataFrame) -> pd.DataFrame:
    return clean_weather(df)
