import pytest
import pandas as pd
import numpy as np

from src.features.build_features import (
    build_time_features,
    build_trend_features,
    build_feature_row,
)
from src.transform.spatial import (
    haversine,
    haversine_vectorized,
    compute_hazard_distances,
    compute_wind_alignment,
)
from src.features.scoring import (
    hazard_type_score,
    proximity_score,
    weather_amplification,
    multi_hazard_bonus,
    compute_risk_score,
    assign_risk_band,
)


class TestSpatial:
    def test_haversine_same_point(self):
        d = haversine(19.0760, 72.8777, 19.0760, 72.8777)
        assert d == 0.0

    def test_haversine_known_distance(self):
        d = haversine(19.0760, 72.8777, 19.1760, 72.8777)
        assert 10 < d < 12  # roughly 11 km

    def test_haversine_vectorized(self):
        lats = np.array([19.0760, 20.0])
        lons = np.array([72.8777, 73.0])
        dists = haversine_vectorized(lats, lons, 19.0760, 72.8777)
        assert dists[0] == 0.0
        assert dists[1] > 100

    def test_compute_hazard_distances(self):
        hazards = pd.DataFrame({
            "latitude": [19.0, 19.5, 20.0],
            "longitude": [72.5, 73.0, 73.5],
            "category": ["Wildfires", "Floods", "Storms"],
        })
        result = compute_hazard_distances(hazards, 19.076, 72.8777, [10, 50, 100])
        assert "hazard_count_10km" in result
        assert "nearest_hazard_distance_km" in result
        assert isinstance(result["hazard_count_10km"], int)

    def test_compute_hazard_distances_empty(self):
        result = compute_hazard_distances(pd.DataFrame(), 19.076, 72.8777, [10, 50, 100])
        assert result["hazard_count_10km"] == 0
        assert result["nearest_hazard_distance_km"] is None

    def test_wind_alignment(self):
        score = compute_wind_alignment(19.0, 72.5, 19.076, 72.8777, 180)
        assert 0 <= score <= 1


class TestFeatureBuilding:
    def test_time_features(self):
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="h")})
        result = build_time_features(df)
        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "weekend" in result.columns

    def test_trend_features(self):
        df = pd.DataFrame({
            "forecast_timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "temperature": range(20, 30),
            "wind_speed": range(10, 20),
            "precipitation": [0] * 10,
        })
        result = build_trend_features(df)
        assert "temp_3h_avg" in result.columns
        assert "temp_1h_change" in result.columns
        assert result["temp_3h_avg"].iloc[0] == 20.0

    def test_build_feature_row(self):
        location = {"latitude": 19.076, "longitude": 72.8777}
        hazards = pd.DataFrame({
            "latitude": [19.0], "longitude": [72.5],
            "category": ["Wildfires"],
        })
        weather = pd.Series({
            "forecast_timestamp": pd.Timestamp("2024-01-01 12:00"),
            "temperature": 30.0, "humidity": 50,
            "wind_speed": 15.0, "wind_direction": 180,
            "precipitation": 0.0, "pressure": 1013, "uv_index": 5,
        })
        config = {
            "risk_scoring": {
                "proximity_bands_km": [10, 50, 100],
            }
        }
        row = build_feature_row(location, hazards, weather, config)
        assert "hazard_count_10km" in row
        assert "temperature" in row
        assert row["temperature"] == 30.0


class TestScoring:
    def test_hazard_type_score(self):
        weights = {"Wildfires": 30, "Floods": 28, "default": 15}
        assert hazard_type_score("Wildfires", weights) == 30
        assert hazard_type_score("Wildfires,Floods", weights) == 30
        assert hazard_type_score("", weights) == 0

    def test_proximity_score(self):
        score = proximity_score(5, [10, 50, 100], [25, 15, 5])
        assert score == 25
        score = proximity_score(30, [10, 50, 100], [25, 15, 5])
        assert score == 15
        score = proximity_score(None, [10, 50, 100], [25, 15, 5])
        assert score == 0

    def test_multi_hazard_bonus(self):
        config = {"risk_scoring": {"multi_hazard_bonus": 5, "max_multi_hazard_bonus": 15}}
        assert multi_hazard_bonus(0, config) == 0
        assert multi_hazard_bonus(1, config) == 0
        assert multi_hazard_bonus(2, config) == 5
        assert multi_hazard_bonus(5, config) == 15

    def test_weather_amplification(self):
        config = {
            "risk_scoring": {
                "weather_amplification": {
                    "high_wind_threshold": 30,
                    "extreme_wind_threshold": 60,
                    "high_wind_score": 10,
                    "extreme_wind_score": 20,
                    "high_humidity_threshold": 80,
                    "high_humidity_score": 5,
                    "heavy_rain_threshold": 10,
                    "heavy_rain_score": 10,
                }
            }
        }
        row = pd.Series({"wind_speed": 70, "humidity": 90, "precipitation": 20})
        score = weather_amplification(row, config)
        assert score >= 20

    def test_compute_risk_score_range(self):
        config = {
            "risk_scoring": {
                "hazard_weights": {"default": 15},
                "proximity_bands_km": [10, 50, 100],
                "proximity_scores": [25, 15, 5],
                "multi_hazard_bonus": 5,
                "max_multi_hazard_bonus": 15,
                "weather_amplification": {
                    "high_wind_threshold": 30,
                    "extreme_wind_threshold": 60,
                    "high_wind_score": 10,
                    "extreme_wind_score": 20,
                    "high_humidity_threshold": 80,
                    "high_humidity_score": 5,
                    "heavy_rain_threshold": 10,
                    "heavy_rain_score": 10,
                },
                "trend": {
                    "worsening_threshold": 5,
                    "worsening_score": 10,
                    "improving_threshold": -5,
                    "improving_score": -5,
                },
            }
        }
        row = pd.Series({
            "hazard_types": "",
            "nearest_hazard_distance_km": None,
            "hazard_count_100km": 0,
            "temp_1h_change": 0,
            "wind_speed": 10,
            "humidity": 50,
            "precipitation": 0,
        })
        score = compute_risk_score(row, config)
        assert 0 <= score <= 100

    def test_assign_risk_band(self):
        bands = [
            {"min": 0, "max": 39, "label": "Safe"},
            {"min": 40, "max": 69, "label": "Caution"},
            {"min": 70, "max": 100, "label": "Avoid"},
        ]
        band, rec = assign_risk_band(20, bands)
        assert band == "Safe"
        assert rec == "Go now"

        band, rec = assign_risk_band(50, bands)
        assert band == "Caution"
        assert rec == "Safe with caution"

        band, rec = assign_risk_band(85, bands)
        assert band == "Avoid"
        assert rec == "Avoid today"
