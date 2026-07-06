import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pandas import isna as pd_isna
from dotenv import load_dotenv

from src.database import get_connection, query, load_feature_table
from src.ingestion.storage_writer import store_parquet

load_dotenv()

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/settings.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def hazard_type_score(hazard_types: str, weights: dict) -> float:
    if not hazard_types or hazard_types == "":
        return 0.0
    types = hazard_types.split(",")
    scores = [weights.get(t.strip(), weights.get("default", 15)) for t in types]
    return float(max(scores))


def proximity_score(
    nearest_dist: float | None,
    bands: list[int],
    band_scores: list[int],
) -> float:
    if nearest_dist is None:
        return 0.0
    for band, score in zip(bands, band_scores):
        if nearest_dist <= band:
            return float(score)
    return 0.0


def weather_amplification(row: pd.Series, config: dict) -> float:
    wa = config["risk_scoring"]["weather_amplification"]
    score = 0.0
    wind = row.get("wind_speed", 0) or 0
    if wind >= wa["extreme_wind_threshold"]:
        score += wa["extreme_wind_score"]
    elif wind >= wa["high_wind_threshold"]:
        score += wa["high_wind_score"]
    humidity = row.get("humidity", 0) or 0
    if humidity >= wa["high_humidity_threshold"]:
        score += wa["high_humidity_score"]
    precip = row.get("precipitation", 0) or 0
    if precip >= wa["heavy_rain_threshold"]:
        score += wa["heavy_rain_score"]
    return score


def multi_hazard_bonus(hazard_count_100km: int, config: dict) -> float:
    if hazard_count_100km <= 1:
        return 0.0
    bonus = config["risk_scoring"]["multi_hazard_bonus"]
    max_bonus = config["risk_scoring"]["max_multi_hazard_bonus"]
    count = max(0, hazard_count_100km - 1)
    return float(min(count * bonus, max_bonus))


def forecast_trend_score(temp_1h_change: float | None, config: dict) -> float:
    if temp_1h_change is None:
        return 0.0
    trend = config["risk_scoring"]["trend"]
    if temp_1h_change >= trend["worsening_threshold"]:
        return float(trend["worsening_score"])
    if temp_1h_change <= trend["improving_threshold"]:
        return float(trend["improving_score"])
    return 0.0


def air_quality_score(aqi: float | None, co: float | None, so2: float | None, config: dict) -> float:
    score = 0.0
    try:
        aq = config["risk_scoring"]["air_quality"]
        if aqi is not None and aqi != 0:
            if aqi >= 4:
                score += float(aq["aqi_high_score"])
            elif aqi >= aq["aqi_threshold"]:
                score += float(aq["aqi_score"])
        if co is not None and co > aq["co_threshold"]:
            score += float(aq["co_score"])
        if so2 is not None and so2 > aq["so2_threshold"]:
            score += float(aq["so2_score"])
    except Exception:
        pass
    return score


def earthquake_magnitude_score(max_mag: float | None, config: dict) -> float:
    if max_mag is None or max_mag == 0:
        return 0.0
    em = config["risk_scoring"]["earthquake_magnitude"]
    if max_mag >= 6:
        return float(em["mag_6_bonus"])
    if max_mag >= 5:
        return float(em["mag_5_bonus"])
    if max_mag >= 4:
        return float(em["mag_4_bonus"])
    return 0.0


def compute_risk_score(row: pd.Series, config: dict) -> float:
    weights = config["risk_scoring"]["hazard_weights"]
    bands = config["risk_scoring"]["proximity_bands_km"]
    band_scores = config["risk_scoring"]["proximity_scores"]

    hz_score = hazard_type_score(row.get("hazard_types", ""), weights)
    prox = proximity_score(row.get("nearest_hazard_distance_km"), bands, band_scores)
    weather_amp = weather_amplification(row, config)
    multi_bonus = multi_hazard_bonus(row.get("hazard_count_100km", 0), config)
    trend = forecast_trend_score(row.get("temp_1h_change"), config)
    aq = air_quality_score(row.get("aqi"), row.get("co"), row.get("so2"), config)
    eq_mag = earthquake_magnitude_score(row.get("max_eq_magnitude"), config)

    score = hz_score + prox + weather_amp + multi_bonus + trend + aq + eq_mag
    return max(0.0, min(100.0, score))


def assign_risk_band(score: float, bands_config: list[dict]) -> tuple[str, str]:
    for band in sorted(bands_config, key=lambda b: b["min"]):
        if band["min"] <= score <= band["max"]:
            label = band["label"]
            if label == "Safe":
                return label, "Go now"
            elif label == "Caution":
                return label, "Safe with caution"
            else:
                return label, "Avoid today"
    return "Unknown", "Check conditions"


def run(feature_df: pd.DataFrame | None = None, location_name: str | None = None) -> pd.DataFrame:
    config = load_config()
    conn = get_connection()

    if feature_df is None:
        feature_df = query(conn, "SELECT * FROM fact_environment_risk")

    if feature_df.empty:
        logger.warning("No feature data to score")
        if conn is not None:
            conn.close()
        return feature_df

    scores = []
    for _, row in feature_df.iterrows():
        cfg = config["risk_scoring"]
        hz_score = hazard_type_score(row.get("hazard_types", ""), cfg["hazard_weights"])
        prox = proximity_score(row.get("nearest_hazard_distance_km"), cfg["proximity_bands_km"], cfg["proximity_scores"])
        weather_amp = weather_amplification(row, config)
        multi_bonus = multi_hazard_bonus(row.get("hazard_count_100km", 0), config)
        trend = forecast_trend_score(row.get("temp_1h_change"), config)
        aq = air_quality_score(row.get("aqi"), row.get("co"), row.get("so2"), config)
        eq_mag = earthquake_magnitude_score(row.get("max_eq_magnitude"), config)
        total = max(0.0, min(100.0, hz_score + prox + weather_amp + multi_bonus + trend + aq + eq_mag))
        band, rec = assign_risk_band(total, cfg["bands"])
        scores.append({
            "risk_score": round(total, 1),
            "risk_band": band,
            "recommendation": rec,
            "component_hazard": round(hz_score, 1),
            "component_proximity": round(prox, 1),
            "component_weather": round(weather_amp, 1),
            "component_trend": round(trend, 1),
            "component_multi_hazard": round(multi_bonus, 1),
            "component_air_quality": round(aq, 1),
        })

    scored_df = feature_df.copy()
    score_df = pd.DataFrame(scores)
    for col in score_df.columns:
        scored_df[col] = score_df[col]

    slug = location_name.lower().replace(" ", "_").replace(",", "") if location_name else None
    path = store_parquet(scored_df, "scored_risk.parquet", subdir="processed", location_slug=slug)
    load_feature_table(conn, scored_df, "fact_environment_risk")

    logger.info("Scored %d rows for '%s', saved to %s", len(scored_df), location_name or "default", path)
    if conn is not None:
        conn.close()
    return scored_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
