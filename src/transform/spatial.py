import logging
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r, lon1_r = radians(lat1), radians(lon1)
    lat2_r, lon2_r = radians(lat2), radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * asin(sqrt(a))


def haversine_vectorized(lats1, lons1, lat2: float, lon2: float) -> np.ndarray:
    lat2_r, lon2_r = radians(lat2), radians(lon2)
    lats1_r = np.radians(lats1)
    lons1_r = np.radians(lons1)
    dlat = lat2_r - lats1_r
    dlon = lon2_r - lons1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lats1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def compute_hazard_distances(
    hazards: pd.DataFrame,
    target_lat: float,
    target_lon: float,
    bands_km: list[int],
) -> dict:
    if hazards.empty:
        return {
            "hazard_count_10km": 0,
            "hazard_count_50km": 0,
            "hazard_count_100km": 0,
            "nearest_hazard_distance_km": None,
            "nearest_hazard_type": None,
            "hazard_types": "",
        }

    distances = haversine_vectorized(
        hazards["latitude"].values,
        hazards["longitude"].values,
        target_lat,
        target_lon,
    )

    result = {
        "hazard_count_10km": int(np.sum(distances <= bands_km[0])),
        "hazard_count_50km": int(np.sum(distances <= bands_km[1])),
        "hazard_count_100km": int(np.sum(distances <= bands_km[2])),
    }

    nearest_idx = np.argmin(distances)
    result["nearest_hazard_distance_km"] = round(float(distances[nearest_idx]), 2)
    result["nearest_hazard_type"] = hazards.iloc[nearest_idx].get("category", "Unknown")

    close_hazards = hazards[distances <= bands_km[2]]
    result["hazard_types"] = ",".join(sorted(close_hazards["category"].unique()))

    return result


def compute_wind_alignment(
    hazard_lat: float,
    hazard_lon: float,
    target_lat: float,
    target_lon: float,
    wind_direction_deg: float,
) -> float:
    """Returns score 0-1: 1 = wind blows from hazard toward user."""
    if wind_direction_deg is None:
        return 0.0
    angle_to_hazard = np.degrees(np.arctan2(
        hazard_lat - target_lat, hazard_lon - target_lon
    )) % 360
    wind_from = wind_direction_deg % 360
    diff = abs(angle_to_hazard - wind_from)
    diff = min(diff, 360 - diff)
    alignment = max(0, 1 - diff / 180)
    return alignment
