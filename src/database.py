import os
import logging
from pathlib import Path

import duckdb
import pandas as pd
import yaml
from dotenv import load_dotenv

from src import database_bq

load_dotenv()

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DUCKDB_PATH", "data/risksense.duckdb")
SQL_DIR = Path("sql")

_USE_BQ = database_bq.is_available()


def _weather_source() -> str:
    try:
        cfg = yaml.safe_load(Path("configs/settings.yaml").read_text())
        return cfg.get("weather", {}).get("provider", "openweathermap")
    except Exception:
        return "openweathermap"


def get_connection():
    if _USE_BQ:
        return None
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = duckdb.connect(DB_PATH)
    conn.execute("SET timezone = 'UTC'")
    return conn


def create_tables(conn=None) -> None:
    if _USE_BQ:
        database_bq.create_tables()
        return
    close = conn is None
    if conn is None:
        conn = get_connection()
    try:
        sql_path = SQL_DIR / "create_tables.sql"
        if sql_path.exists():
            sql = sql_path.read_text()
            conn.execute(sql)
            logger.info("Created tables from %s", sql_path)
        sql_path = SQL_DIR / "views.sql"
        if sql_path.exists():
            sql = sql_path.read_text()
            conn.execute(sql)
            logger.info("Created views from %s", sql_path)
        _migrate_schema(conn)
    finally:
        if close:
            conn.close()


def _migrate_schema(conn: duckdb.DuckDBPyConnection) -> None:
    migrations = {
        "fact_environment_risk": [
            ("component_air_quality", "DOUBLE DEFAULT 0"),
            ("aqi", "INTEGER"),
            ("co", "DOUBLE"),
            ("so2", "DOUBLE"),
            ("max_eq_magnitude", "DOUBLE DEFAULT 0"),
        ],
        "raw_weather": [
            ("wind_gust", "DOUBLE"),
        ],
    }
    for table, columns in migrations.items():
        try:
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for col_name, col_type in columns:
                if col_name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                    logger.info("Migrated: added %s to %s", col_name, table)
        except Exception as e:
            logger.debug("Schema migration for %s: %s", table, e)


def load_eonet_events(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    if _USE_BQ:
        return database_bq.load_eonet_events(df)
    cols = ", ".join(df.columns)
    placeholders = ", ".join([f"df.\"{c}\"" for c in df.columns])
    conn.execute("DELETE FROM raw_eonet_events WHERE 1=1")
    conn.execute(f"INSERT INTO raw_eonet_events ({cols}) SELECT {placeholders} FROM df")
    logger.info("Loaded %d EONET events into DuckDB", len(df))
    return len(df)


def load_weather(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    if _USE_BQ:
        return database_bq.load_weather(df)
    df = df.copy()
    df["location_id"] = 1
    df["source_name"] = _weather_source()
    cols = ", ".join(df.columns)
    placeholders = ", ".join([f"df.\"{c}\"" for c in df.columns])
    conn.execute("DELETE FROM raw_weather WHERE 1=1")
    conn.execute(f"INSERT INTO raw_weather ({cols}) SELECT {placeholders} FROM df")
    logger.info("Loaded %d weather rows into DuckDB", len(df))
    return len(df)


def load_feature_table(conn, df: pd.DataFrame, table_name: str) -> int:
    if df.empty:
        return 0
    if _USE_BQ:
        return database_bq.load_feature_table(df, table_name)
    cols = ", ".join(df.columns)
    placeholders = ", ".join([f"df.\"{c}\"" for c in df.columns])
    conn.execute(f"DELETE FROM {table_name} WHERE 1=1")
    conn.execute(f"INSERT INTO {table_name} ({cols}) SELECT {placeholders} FROM df")
    logger.info("Loaded %d rows into %s", len(df), table_name)
    return len(df)


def load_air_quality(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    if _USE_BQ:
        return database_bq.load_air_quality(df)
    cols = ", ".join(df.columns)
    placeholders = ", ".join([f"df.\"{c}\"" for c in df.columns])
    conn.execute("DELETE FROM raw_air_quality WHERE 1=1")
    conn.execute(f"INSERT INTO raw_air_quality ({cols}) SELECT {placeholders} FROM df")
    logger.info("Loaded %d air quality rows into DuckDB", len(df))
    return len(df)


def load_earthquakes(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    if _USE_BQ:
        return database_bq.load_earthquakes(df)
    cols = ", ".join(df.columns)
    placeholders = ", ".join([f"df.\"{c}\"" for c in df.columns])
    conn.execute("DELETE FROM raw_earthquakes WHERE 1=1")
    conn.execute(f"INSERT INTO raw_earthquakes ({cols}) SELECT {placeholders} FROM df")
    logger.info("Loaded %d earthquake rows into DuckDB", len(df))
    return len(df)


def load_explanations(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    if _USE_BQ:
        return database_bq.load_explanations(df)
    cols = ", ".join(df.columns)
    placeholders = ", ".join([f"df.\"{c}\"" for c in df.columns])
    conn.execute("DELETE FROM raw_explanations WHERE 1=1")
    conn.execute(f"INSERT INTO raw_explanations ({cols}) SELECT {placeholders} FROM df")
    logger.info("Loaded %d explanation rows into DuckDB", len(df))
    return len(df)


def query(conn, sql: str) -> pd.DataFrame:
    if _USE_BQ:
        return database_bq.query(sql)
    return conn.execute(sql).fetchdf()


def load_all_raw(conn, raw_dir: str = "data/raw") -> tuple[int, int]:
    if _USE_BQ:
        logger.info("BigQuery mode — load_all_raw delegates to fetch pipeline")
        return 0, 0
    base = Path(raw_dir)
    eonet_count = 0
    weather_count = 0
    eonet_files = sorted(base.glob("eonet_events_*.parquet"))
    if eonet_files:
        df = pd.read_parquet(eonet_files[-1])
        eonet_count = load_eonet_events(conn, df)
    weather_files = sorted(base.glob("weather_*.parquet"))
    if weather_files:
        df = pd.read_parquet(weather_files[-1])
        weather_count = load_weather(conn, df)
    return eonet_count, weather_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    conn = get_connection()
    create_tables(conn)
    e, w = load_all_raw(conn)
    logger.info("Done: %d EONET events, %d weather rows loaded", e, w)
    if conn:
        conn.close()
