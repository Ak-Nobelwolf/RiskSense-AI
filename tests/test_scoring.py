import pytest
import pandas as pd

from src.features.scoring import compute_risk_score, assign_risk_band, air_quality_score
from src.explain.rules import build_explanation


class TestEndToEndScenarios:
    def _make_config(self, overrides: dict | None = None) -> dict:
        cfg = {
            "hazard_weights": {"default": 15},
            "proximity_bands_km": [10, 50, 100],
            "proximity_scores": [25, 15, 5],
            "multi_hazard_bonus": 5,
            "max_multi_hazard_bonus": 15,
            "weather_amplification": {
                "high_wind_threshold": 30, "extreme_wind_threshold": 60,
                "high_wind_score": 10, "extreme_wind_score": 20,
                "high_humidity_threshold": 80, "high_humidity_score": 5,
                "heavy_rain_threshold": 10, "heavy_rain_score": 10,
            },
            "trend": {
                "worsening_threshold": 5, "worsening_score": 10,
                "improving_threshold": -5, "improving_score": -5,
            },
            "air_quality": {"aqi_threshold": 3, "aqi_score": 15, "aqi_high_score": 25, "co_threshold": 10, "co_score": 5, "so2_threshold": 20, "so2_score": 5},
            "earthquake_magnitude": {"mag_4_bonus": 5, "mag_5_bonus": 10, "mag_6_bonus": 15},
        }
        if overrides:
            cfg.update(overrides)
        return {"risk_scoring": cfg}

    def test_no_hazards_safe_weather(self):
        config = self._make_config()
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
        assert score <= 39

    def test_wildfire_nearby(self):
        config = self._make_config({"hazard_weights": {"Wildfires": 30, "default": 15}})
        row = pd.Series({
            "hazard_types": "Wildfires",
            "nearest_hazard_distance_km": 8,
            "hazard_count_100km": 2,
            "temp_1h_change": 3,
            "wind_speed": 35,
            "humidity": 50,
            "precipitation": 0,
        })
        score = compute_risk_score(row, config)
        assert score >= 70

    def test_moderate_conditions(self):
        config = self._make_config({"hazard_weights": {"Floods": 28, "default": 15}})
        row = pd.Series({
            "hazard_types": "Floods",
            "nearest_hazard_distance_km": 60,
            "hazard_count_100km": 1,
            "temp_1h_change": 0,
            "wind_speed": 15,
            "humidity": 60,
            "precipitation": 2,
        })
        score = compute_risk_score(row, config)
        assert 0 <= score <= 39  # flood 60km away + no other hazards = safe

    def test_band_mapping(self):
        bands = [
            {"min": 0, "max": 39, "label": "Safe"},
            {"min": 40, "max": 69, "label": "Caution"},
            {"min": 70, "max": 100, "label": "Avoid"},
        ]
        assert assign_risk_band(0, bands)[0] == "Safe"
        assert assign_risk_band(39, bands)[0] == "Safe"
        assert assign_risk_band(40, bands)[0] == "Caution"
        assert assign_risk_band(69, bands)[0] == "Caution"
        assert assign_risk_band(70, bands)[0] == "Avoid"
        assert assign_risk_band(100, bands)[0] == "Avoid"


class TestAirQualityScore:
    def _aq_config(self):
        return {"risk_scoring": {"air_quality": {"aqi_threshold": 3, "aqi_score": 15, "aqi_high_score": 25, "co_threshold": 10, "co_score": 5, "so2_threshold": 20, "so2_score": 5}}}

    def test_aq_none(self):
        cfg = self._aq_config()
        assert air_quality_score(None, None, None, cfg) == 0.0

    def test_aq_zero(self):
        cfg = self._aq_config()
        assert air_quality_score(0, None, None, cfg) == 0.0

    def test_aq_nan(self):
        cfg = self._aq_config()
        assert air_quality_score(float("nan"), None, None, cfg) == 0.0

    def test_aq_below_threshold(self):
        cfg = self._aq_config()
        assert air_quality_score(2, None, None, cfg) == 0.0

    def test_aq_at_threshold(self):
        cfg = self._aq_config()
        assert air_quality_score(3, None, None, cfg) == 15.0

    def test_aq_high(self):
        cfg = self._aq_config()
        assert air_quality_score(4, None, None, cfg) == 25.0

    def test_aq_very_high(self):
        cfg = self._aq_config()
        assert air_quality_score(5, None, None, cfg) == 25.0

    def test_aq_integration(self):
        """verify aqi feeds through compute_risk_score"""
        cfg = {
            "risk_scoring": {
                "hazard_weights": {"default": 15},
                "proximity_bands_km": [10, 50, 100],
                "proximity_scores": [25, 15, 5],
                "multi_hazard_bonus": 5,
                "max_multi_hazard_bonus": 15,
                "weather_amplification": {
                    "high_wind_threshold": 30, "extreme_wind_threshold": 60,
                    "high_wind_score": 10, "extreme_wind_score": 20,
                    "high_humidity_threshold": 80, "high_humidity_score": 5,
                    "heavy_rain_threshold": 10, "heavy_rain_score": 10,
                },
                "trend": {
                    "worsening_threshold": 5, "worsening_score": 10,
                    "improving_threshold": -5, "improving_score": -5,
                },
                "air_quality": {"aqi_threshold": 3, "aqi_score": 15, "aqi_high_score": 25, "co_threshold": 10, "co_score": 5, "so2_threshold": 20, "so2_score": 5},
                "earthquake_magnitude": {"mag_4_bonus": 5, "mag_5_bonus": 10, "mag_6_bonus": 15},
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
            "aqi": 4,
            "co": None,
            "so2": None,
            "max_eq_magnitude": 0,
        })
        score = compute_risk_score(row, cfg)
        # aqi=4 adds 25, other factors add 0 → score = 25
        assert 24 <= score <= 26


class TestExplanation:
    def test_explain_safe(self):
        row = {
            "risk_score": 20,
            "risk_band": "Safe",
            "recommendation": "Go now",
            "hazard_count_100km": 0,
            "nearest_hazard_distance_km": None,
            "nearest_hazard_type": "None",
            "hazard_types": "",
            "wind_speed": 10,
            "temp_1h_change": 0,
        }
        result = build_explanation(row)
        assert result["risk_band"] == "Safe"
        assert "safe" in result["summary"].lower()

    def test_explain_avoid(self):
        row = {
            "risk_score": 85,
            "risk_band": "Avoid",
            "recommendation": "Avoid today",
            "hazard_count_100km": 3,
            "nearest_hazard_distance_km": 5,
            "nearest_hazard_type": "Wildfires",
            "hazard_types": "Wildfires,Floods",
            "wind_speed": 50,
            "temp_1h_change": 6,
        }
        result = build_explanation(row)
        assert result["risk_band"] == "Avoid"
        assert "avoid" in result["summary"].lower()
        assert len(result["top_reasons"]) > 0

    def test_explain_reasons_count(self):
        row = {
            "risk_score": 60,
            "risk_band": "Caution",
            "recommendation": "Safe with caution",
            "hazard_count_100km": 2,
            "nearest_hazard_distance_km": 30,
            "nearest_hazard_type": "Floods",
            "hazard_types": "Floods",
            "wind_speed": 35,
            "temp_1h_change": 2,
        }
        result = build_explanation(row)
        assert len(result["top_reasons"]) <= 3
        assert isinstance(result["confidence"], float)
