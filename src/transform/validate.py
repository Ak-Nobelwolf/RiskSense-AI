import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_coordinates(df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude") -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=[lat_col, lon_col])
    df = df[(df[lat_col].between(-90, 90)) & (df[lon_col].between(-180, 180))]
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with invalid coordinates", dropped)
    return df


def validate_columns(df: pd.DataFrame, required: list[str]) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        return False
    return True


def validate_timestamps(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[ts_col])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with invalid timestamps in %s", dropped, ts_col)
    return df


def validate_risk_score(score: float) -> bool:
    return 0.0 <= score <= 100.0
