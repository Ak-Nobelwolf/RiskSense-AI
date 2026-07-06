from __future__ import annotations

import logging
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BQ_AVAILABLE = bool(os.getenv("GCP_PROJECT_ID")) and (
    bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")) or bool(os.getenv("K_SERVICE"))
)

_client = None

_TABLES = [
    "raw_eonet_events",
    "raw_weather",
    "raw_air_quality",
    "raw_earthquakes",
    "raw_explanations",
    "dim_location",
    "fact_environment_risk",
]


def _get_client():
    global _client
    if _client is None and BQ_AVAILABLE:
        from google.cloud import bigquery
        _client = bigquery.Client()
    return _client


def _dataset_ref() -> str | None:
    project = os.getenv("GCP_PROJECT_ID")
    dataset = os.getenv("BIGQUERY_DATASET", "risksense")
    if not project:
        return None
    return f"{project}.{dataset}"


def _t(table: str) -> str:
    ds = _dataset_ref()
    return f"`{ds}.{table}`"


def _tid(table: str) -> str:
    ds = _dataset_ref()
    return f"{ds}.{table}"


def is_available() -> bool:
    return BQ_AVAILABLE and _get_client() is not None


def create_tables() -> None:
    if not is_available():
        logger.warning("BigQuery not available, skipping create_tables")
        return
    client = _get_client()
    ds = _dataset_ref()
    if not ds:
        return

    ddl = f"""
CREATE TABLE IF NOT EXISTS {_t("raw_eonet_events")} (
    event_id STRING,
    category STRING,
    title STRING,
    status STRING,
    event_timestamp TIMESTAMP,
    latitude FLOAT64,
    longitude FLOAT64
);

CREATE TABLE IF NOT EXISTS {_t("raw_weather")} (
    forecast_timestamp TIMESTAMP,
    temperature FLOAT64,
    humidity FLOAT64,
    wind_speed FLOAT64,
    wind_gust FLOAT64,
    wind_direction FLOAT64,
    precipitation FLOAT64,
    pressure FLOAT64,
    uv_index FLOAT64,
    condition_text STRING,
    location_id INT64 DEFAULT 1,
    source_name STRING DEFAULT 'openweathermap',
    ingestion_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS {_t("raw_air_quality")} (
    timestamp TIMESTAMP,
    aqi INT64,
    co FLOAT64,
    so2 FLOAT64,
    pm25 FLOAT64,
    pm10 FLOAT64,
    o3 FLOAT64,
    no2 FLOAT64,
    location_id INT64 DEFAULT 1,
    source_name STRING DEFAULT 'openweathermap',
    ingestion_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS {_t("raw_earthquakes")} (
    id STRING,
    title STRING,
    magnitude FLOAT64,
    place STRING,
    time TIMESTAMP,
    latitude FLOAT64,
    longitude FLOAT64,
    depth FLOAT64,
    location_id INT64 DEFAULT 1,
    ingestion_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS {_t("raw_explanations")} (
    location_id INT64 DEFAULT 1,
    timestamp TIMESTAMP,
    explanation_text STRING,
    provider STRING DEFAULT 'rule_based',
    ingestion_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS {_t("dim_location")} (
    location_id INT64,
    location_name STRING,
    latitude FLOAT64,
    longitude FLOAT64,
    city STRING,
    region STRING,
    country STRING
);

CREATE TABLE IF NOT EXISTS {_t("fact_environment_risk")} (
    location_id INT64 DEFAULT 1,
    timestamp TIMESTAMP,
    hazard_count_10km INT64 DEFAULT 0,
    hazard_count_50km INT64 DEFAULT 0,
    hazard_count_100km INT64 DEFAULT 0,
    nearest_hazard_distance_km FLOAT64,
    nearest_hazard_type STRING,
    hazard_types STRING,
    hazard_severity_score FLOAT64 DEFAULT 0,
    wind_alignment_score FLOAT64 DEFAULT 0,
    trend_score FLOAT64 DEFAULT 0,
    temperature FLOAT64,
    humidity FLOAT64,
    wind_speed FLOAT64,
    wind_direction FLOAT64,
    precipitation FLOAT64,
    pressure FLOAT64,
    uv_index FLOAT64,
    aqi INT64,
    co FLOAT64,
    so2 FLOAT64,
    max_eq_magnitude FLOAT64 DEFAULT 0,
    hour INT64,
    day_of_week INT64,
    weekend INT64,
    temp_3h_avg FLOAT64,
    temp_6h_avg FLOAT64,
    temp_1h_change FLOAT64,
    temp_3h_change FLOAT64,
    wind_3h_avg FLOAT64,
    wind_1h_change FLOAT64,
    precip_3h_sum FLOAT64,
    risk_score FLOAT64,
    risk_band STRING,
    recommendation STRING,
    component_hazard FLOAT64 DEFAULT 0,
    component_proximity FLOAT64 DEFAULT 0,
    component_weather FLOAT64 DEFAULT 0,
    component_trend FLOAT64 DEFAULT 0,
    component_multi_hazard FLOAT64 DEFAULT 0,
    component_air_quality FLOAT64 DEFAULT 0,
    predicted_risk_6h FLOAT64,
    predicted_risk_12h FLOAT64,
    confidence_score FLOAT64 DEFAULT 0.8,
    explanation_text STRING
);

-- Insert default Mumbai location (idempotent)
MERGE INTO {_t("dim_location")} t
USING (SELECT 1 AS location_id) s
ON t.location_id = s.location_id
WHEN NOT MATCHED THEN
    INSERT (location_id, location_name, latitude, longitude, city, region, country)
    VALUES (1, 'Mumbai Central', 19.0760, 72.8777, 'Mumbai', 'Maharashtra', 'India');

CREATE OR REPLACE VIEW `{ds}.v_current_risk` AS
SELECT
    dl.location_name,
    dl.city,
    dl.region,
    fer.timestamp,
    fer.risk_score,
    fer.risk_band,
    fer.recommendation,
    fer.hazard_count_10km,
    fer.hazard_count_50km,
    fer.nearest_hazard_distance_km,
    fer.nearest_hazard_type,
    fer.explanation_text
FROM {_t("fact_environment_risk")} fer
JOIN {_t("dim_location")} dl ON fer.location_id = dl.location_id
WHERE fer.timestamp = (SELECT MAX(timestamp) FROM {_t("fact_environment_risk")});

CREATE OR REPLACE VIEW `{ds}.v_active_hazards` AS
SELECT
    event_id,
    category,
    title,
    latitude,
    longitude,
    event_timestamp
FROM {_t("raw_eonet_events")}
WHERE status = 'open';

CREATE OR REPLACE VIEW `{ds}.v_risk_trend` AS
SELECT
    fer.location_id,
    fer.timestamp,
    fer.risk_score,
    fer.risk_band,
    fer.predicted_risk_6h,
    fer.predicted_risk_12h
FROM {_t("fact_environment_risk")} fer
ORDER BY fer.timestamp;
"""
    for statement in ddl.split(";"):
        statement = statement.strip()
        if not statement:
            continue
        try:
            client.query(statement).result()
        except Exception as e:
            logger.warning("BigQuery DDL statement failed: %s\nSQL: %s...", e, statement[:80])


def _load_job_config():
    from google.cloud.bigquery import LoadJobConfig
    return LoadJobConfig(write_disposition="WRITE_TRUNCATE")


def load_eonet_events(df: pd.DataFrame) -> int:
    if not is_available() or df.empty:
        return 0
    client = _get_client()
    table_id = _tid("raw_eonet_events")
    job = client.load_table_from_dataframe(df, table_id, job_config=_load_job_config())
    job.result()
    logger.info("Loaded %d EONET events into BigQuery", len(df))
    return len(df)


def load_weather(df: pd.DataFrame) -> int:
    if not is_available() or df.empty:
        return 0
    client = _get_client()
    table_id = _tid("raw_weather")
    df = df.copy()
    df["location_id"] = 1
    job = client.load_table_from_dataframe(df, table_id, job_config=_load_job_config())
    job.result()
    logger.info("Loaded %d weather rows into BigQuery", len(df))
    return len(df)


def load_feature_table(df: pd.DataFrame, table_name: str) -> int:
    if not is_available() or df.empty:
        return 0
    client = _get_client()
    table_id = _tid(table_name)
    job = client.load_table_from_dataframe(df, table_id, job_config=_load_job_config())
    job.result()
    logger.info("Loaded %d rows into BigQuery %s", len(df), table_name)
    return len(df)


def load_air_quality(df: pd.DataFrame) -> int:
    if not is_available() or df.empty:
        return 0
    client = _get_client()
    table_id = _tid("raw_air_quality")
    df = df.copy()
    df["location_id"] = 1
    df["source_name"] = "openweathermap"
    job = client.load_table_from_dataframe(df, table_id, job_config=_load_job_config())
    job.result()
    logger.info("Loaded %d air quality rows into BigQuery", len(df))
    return len(df)


def load_earthquakes(df: pd.DataFrame) -> int:
    if not is_available() or df.empty:
        return 0
    client = _get_client()
    table_id = _tid("raw_earthquakes")
    df = df.copy()
    df["location_id"] = 1
    job = client.load_table_from_dataframe(df, table_id, job_config=_load_job_config())
    job.result()
    logger.info("Loaded %d earthquake rows into BigQuery", len(df))
    return len(df)


def load_explanations(df: pd.DataFrame) -> int:
    if not is_available() or df.empty:
        return 0
    client = _get_client()
    table_id = _tid("raw_explanations")
    df = df.copy()
    df["location_id"] = 1
    job = client.load_table_from_dataframe(df, table_id, job_config=_load_job_config())
    job.result()
    logger.info("Loaded %d explanation rows into BigQuery", len(df))
    return len(df)


def query(sql: str) -> pd.DataFrame:
    if not is_available():
        return pd.DataFrame()
    client = _get_client()
    sql_bq = _qualify_tables(sql)
    try:
        df = client.query(sql_bq).result().to_dataframe()
        return df
    except Exception as e:
        logger.warning("BigQuery query failed: %s — SQL: %s", e, sql_bq)
        return pd.DataFrame()


def _qualify_tables(sql: str) -> str:
    ds = _dataset_ref()
    if not ds:
        return sql
    for table in _TABLES:
        sql = sql.replace(f"FROM {table}", f"FROM `{ds}.{table}`")
        sql = sql.replace(f"JOIN {table}", f"JOIN `{ds}.{table}`")
        sql = sql.replace(f"FROM {table} ", f"FROM `{ds}.{table}` ")
        sql = sql.replace(f"JOIN {table} ", f"JOIN `{ds}.{table}` ")
    sql = sql.replace('"', "'")
    return sql
