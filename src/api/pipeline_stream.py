from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from collections.abc import Generator


def _read_parquet(pattern: str, subdir: str = "raw") -> pd.DataFrame:
    files = sorted(Path(f"data/{subdir}").glob(pattern))
    return pd.read_parquet(files[-1]) if files else pd.DataFrame()


def run_pipeline_with_progress(
    lat: float, lon: float, location_name: str, region: dict,
) -> Generator[dict, None, None]:
    yield {"step": "Initializing\u2026", "status": "running"}

    slug = location_name.lower().replace(" ", "_").replace(",", "")
    for pattern in [f"data/processed/*__{slug}.parquet", f"data/raw/*__{slug}.parquet"]:
        for f in Path().glob(pattern):
            f.unlink()
            logger = __import__('logging').getLogger(__name__)
            logger.info("Deleted stale: %s", f)

    from src.database import get_connection, create_tables, load_eonet_events, load_weather, load_air_quality, load_earthquakes, load_explanations as db_load_explanations

    conn = get_connection()
    create_tables(conn)

    yield {"step": "Loading hazard events\u2026", "status": "running"}
    from src.ingestion.eonet_fetcher import run as ingest_eonet
    eonet_df = ingest_eonet()
    load_eonet_events(conn, eonet_df)

    yield {"step": "Fetching current weather\u2026", "status": "running"}
    from src.ingestion.weather_fetcher import run as ingest_weather
    ingest_weather(lat=lat, lon=lon, location_name=location_name, force_refresh=True)

    yield {"step": "Fetching air quality\u2026", "status": "running"}
    from src.ingestion.air_quality_fetcher import run as ingest_aq
    ingest_aq(lat=lat, lon=lon, location_name=location_name, force_refresh=True)

    yield {"step": "Fetching seismic activity\u2026", "status": "running"}
    from src.ingestion.usgs_fetcher import run as ingest_usgs
    ingest_usgs(lat=lat, lon=lon, location_name=location_name, force_refresh=True, max_radius_km=300)

    # Load ingested data to database (read from parquet, not BQ — BQ is still empty at this stage)
    weather_df = _read_parquet(f"weather__{slug}.parquet")
    if not weather_df.empty:
        load_weather(conn, weather_df)
    aq_df = _read_parquet(f"air_quality__{slug}.parquet")
    if not aq_df.empty:
        load_air_quality(conn, aq_df)
    eq_df = _read_parquet(f"earthquakes__{slug}.parquet")
    if not eq_df.empty:
        load_earthquakes(conn, eq_df)

    if conn is not None:
        conn.close()

    yield {"step": "Analyzing patterns\u2026", "status": "running"}
    from src.features.build_features import run as build_features
    build_features(location=region)

    yield {"step": "Scoring environmental risk\u2026", "status": "running"}
    from src.features.scoring import run as score_risk
    score_risk(location_name=location_name)

    yield {"step": "Generating forecast\u2026", "status": "running"}
    from src.model.predict import run as predict_risk
    predict_risk(location_name=location_name)

    yield {"step": "Preparing summary\u2026", "status": "running"}
    from src.explain.prompt_builder import run as explain_risk
    explain_risk(location_name=location_name)

    # Load explanations to database (read from parquet, not BQ)
    conn2 = get_connection()
    expl_df = _read_parquet(f"explanations__{slug}.parquet", subdir="processed")
    if not expl_df.empty:
        db_load_explanations(conn2, expl_df)
    if conn2 is not None:
        conn2.close()

    yield {"step": "Complete \u2713", "status": "complete"}
