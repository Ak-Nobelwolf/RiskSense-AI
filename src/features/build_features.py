import logging
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.database import get_connection, query, load_feature_table, create_tables
from src.transform.spatial import compute_hazard_distances, compute_wind_alignment, haversine_vectorized
from src.transform.validate import validate_coordinates
from src.dashboard.utils import load_config, load_weather_raw, load_air_quality_raw, load_earthquake_raw
from src.ingestion.storage_writer import store_parquet

load_dotenv()

logger = logging.getLogger(__name__)


def build_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df["hour"] = df[ts_col].dt.hour
    df["day_of_week"] = df[ts_col].dt.dayofweek
    df["weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def build_trend_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    if weather_df.empty:
        return weather_df
    df = weather_df.sort_values("forecast_timestamp").copy()
    df["temp_3h_avg"] = df["temperature"].rolling(3, min_periods=1).mean()
    df["temp_6h_avg"] = df["temperature"].rolling(6, min_periods=1).mean()
    df["temp_1h_change"] = df["temperature"].diff(1)
    df["temp_3h_change"] = df["temperature"].diff(3)
    df["wind_3h_avg"] = df["wind_speed"].rolling(3, min_periods=1).mean()
    df["wind_1h_change"] = df["wind_speed"].diff(1)
    df["precip_3h_sum"] = df["precipitation"].rolling(3, min_periods=1).sum()
    return df


def build_feature_row(
    location: dict,
    hazards: pd.DataFrame,
    weather_row: pd.Series,
    config: dict,
    aqi: float | None = None,
    co: float | None = None,
    so2: float | None = None,
    max_eq_magnitude: float = 0.0,
) -> dict:
    target_lat = location["latitude"]
    target_lon = location["longitude"]
    bands = config["risk_scoring"]["proximity_bands_km"]

    spatial = compute_hazard_distances(hazards, target_lat, target_lon, bands)

    wind_align = 0.0
    if spatial["nearest_hazard_distance_km"] is not None:
        wind_align = compute_wind_alignment(
            hazards.iloc[0]["latitude"] if not hazards.empty else target_lat,
            hazards.iloc[0]["longitude"] if not hazards.empty else target_lon,
            target_lat,
            target_lon,
            weather_row.get("wind_direction", 0),
        )

    row = {
        "timestamp": weather_row.get("forecast_timestamp"),
        "hazard_count_10km": spatial["hazard_count_10km"],
        "hazard_count_50km": spatial["hazard_count_50km"],
        "hazard_count_100km": spatial["hazard_count_100km"],
        "nearest_hazard_distance_km": spatial["nearest_hazard_distance_km"],
        "nearest_hazard_type": spatial["nearest_hazard_type"] or "None",
        "hazard_types": spatial["hazard_types"],
        "temperature": weather_row.get("temperature"),
        "humidity": weather_row.get("humidity"),
        "wind_speed": weather_row.get("wind_speed"),
        "wind_direction": weather_row.get("wind_direction"),
        "precipitation": weather_row.get("precipitation"),
        "pressure": weather_row.get("pressure"),
        "uv_index": weather_row.get("uv_index"),
        "wind_alignment_score": round(wind_align * 100, 2),
        "aqi": aqi,
        "co": co,
        "so2": so2,
        "max_eq_magnitude": max_eq_magnitude,
    }
    return row


def run(location: dict | None = None) -> pd.DataFrame:
    config = load_config()
    if location is None:
        location = config["region"]

    conn = get_connection()
    hazards = query(conn, "SELECT * FROM raw_eonet_events WHERE status IN ('open', 'unknown')")
    if conn is not None:
        conn.close()

    location_name = location.get("name") or location.get("city", "unknown")

    # Merge USGS earthquakes into hazard data for distance computation
    eq_df = load_earthquake_raw(location=location_name)
    max_eq_magnitude = 0.0
    if not eq_df.empty:
        eq_hazards = eq_df[["latitude", "longitude", "title", "category", "status", "magnitude"]].copy()
        eq_hazards["status"] = "open"
        # Compute max earthquake magnitude within 100km
        eq_dists = haversine_vectorized(eq_df["latitude"].values, eq_df["longitude"].values,
                                         location["latitude"], location["longitude"])
        eq_close = eq_df[eq_dists <= 100]
        max_eq_magnitude = float(eq_close["magnitude"].max()) if not eq_close.empty else 0.0
        hazards = pd.concat([hazards, eq_hazards], ignore_index=True)
        logger.info("Merged %d earthquakes into hazard data for '%s', max mag=%.1f", len(eq_hazards), location_name, max_eq_magnitude)

    weather = load_weather_raw(location=location_name)

    hazards_clean = validate_coordinates(hazards)
    weather_clean = weather.dropna(subset=["forecast_timestamp"]).copy()

    if weather_clean.empty:
        logger.warning("No weather data available for '%s'", location_name)
        return pd.DataFrame()

    weather_features = build_trend_features(weather_clean)

    # Merge AQI data: forward-fill nearest AQI value for each weather timestamp
    aq_df = load_air_quality_raw(location=location_name)
    if not aq_df.empty:
        aq_sorted = aq_df[["timestamp", "aqi", "co", "so2"]].sort_values("timestamp")
        wf = weather_features.reset_index(drop=True).copy()
        wf["_merge_key"] = wf["forecast_timestamp"]
        aq_sorted["_merge_key"] = aq_sorted["timestamp"]
        merged = pd.merge_asof(
            wf.sort_values("forecast_timestamp"),
            aq_sorted.sort_values("timestamp"),
            on="_merge_key",
            direction="nearest",
            tolerance=pd.Timedelta("3h"),
        )
        weather_features["aqi"] = merged["aqi"]
        weather_features["co"] = merged["co"]
        weather_features["so2"] = merged["so2"]
    else:
        logger.info("No AQI data for '%s' - aqi/co/so2 will be None", location_name)

    rows = []
    for _, wrow in weather_features.iterrows():
        aqi_val = wrow.get("aqi")
        if pd.notna(aqi_val):
            aqi_val = int(aqi_val)
        co_val = wrow.get("co")
        co_val = None if pd.isna(co_val) else float(co_val)
        so2_val = wrow.get("so2")
        so2_val = None if pd.isna(so2_val) else float(so2_val)
        rows.append(build_feature_row(
            location, hazards_clean, wrow, config,
            aqi=aqi_val, co=co_val, so2=so2_val,
            max_eq_magnitude=max_eq_magnitude,
        ))

    feature_df = pd.DataFrame(rows)
    # Convert NaN aqi to None to avoid pd.NA issues downstream
    if "aqi" in feature_df.columns:
        feature_df["aqi"] = feature_df["aqi"].apply(lambda x: int(x) if pd.notna(x) else None)
    feature_df = build_time_features(feature_df)

    slug = location_name.lower().replace(" ", "_").replace(",", "")
    path = store_parquet(feature_df, "features.parquet", subdir="processed", location_slug=slug)

    conn2 = get_connection()
    create_tables(conn2)
    load_feature_table(conn2, feature_df, "fact_environment_risk")
    if conn2 is not None:
        conn2.close()

    logger.info("Built %d feature rows for '%s', saved to %s", len(feature_df), location_name, path)
    return feature_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
