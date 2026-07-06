import json
import pytest
import pandas as pd
import numpy as np

from src.ingestion.eonet_fetcher import normalize_event
from src.ingestion.weather_fetcher import normalize_owm_forecast
from src.transform.clean import clean_eonet_events, clean_weather
from src.transform.validate import validate_coordinates, validate_risk_score


class TestEonetNormalization:
    def test_normalize_valid_event(self):
        event = {
            "id": "EONET_123",
            "title": "Test Wildfire",
            "categories": [{"title": "Wildfires"}],
            "status": "open",
            "geometry": [{"date": "2024-01-01T00:00:00Z", "coordinates": [72.8777, 19.0760]}],
            "sources": [{"url": "http://example.com", "id": "test"}],
        }
        result = normalize_event(event)
        assert result is not None
        assert result["event_id"] == "EONET_123"
        assert result["category"] == "Wildfires"
        assert result["latitude"] == 19.0760
        assert result["longitude"] == 72.8777

    def test_normalize_event_no_geometry(self):
        event = {
            "id": "EONET_456",
            "title": "No Geometry",
            "categories": [{"title": "Floods"}],
            "status": "open",
            "geometry": [],
            "sources": [],
        }
        result = normalize_event(event)
        assert result is None

    def test_normalize_event_wrong_coords(self):
        event = {
            "id": "EONET_789",
            "title": "Bad Coords",
            "categories": [{"title": "Storms"}],
            "status": "open",
            "geometry": [{"date": "2024-01-01T00:00:00Z", "coordinates": [1]}],
            "sources": [],
        }
        result = normalize_event(event)
        assert result is None


class TestWeatherNormalization:
    def test_normalize_forecast(self):
        data = {
            "list": [
                {
                    "dt": 1704067200,
                    "main": {"temp": 25.0, "humidity": 60, "pressure": 1013.0},
                    "weather": [{"description": "clear sky"}],
                    "wind": {"speed": 15.0, "deg": 180},
                }
            ],
            "city": {"name": "Mumbai"},
        }
        df = normalize_owm_forecast(data, 19.076, 72.8777)
        assert not df.empty
        assert df.iloc[0]["temperature"] == 25.0
        assert df.iloc[0]["humidity"] == 60
        assert df.iloc[0]["wind_speed"] == 15.0

    def test_normalize_forecast_empty(self):
        data = {"list": [], "city": {}}
        df = normalize_owm_forecast(data, 0, 0)
        assert df.empty


class TestCleaning:
    def test_clean_eonet(self):
        df = pd.DataFrame({
            "event_id": ["a", "a", "b"],
            "category": ["Wildfires", "Wildfires", "Floods"],
            "title": ["Fire 1", "Fire 1", "Flood 1"],
            "latitude": [19.0, 19.0, 20.0],
            "longitude": [72.0, 72.0, 73.0],
            "event_timestamp": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "status": ["open", "open", "open"],
        })
        result = clean_eonet_events(df)
        assert len(result) == 2  # dedup removes one

    def test_clean_eonet_bad_coords(self):
        df = pd.DataFrame({
            "event_id": ["a", "b"],
            "category": ["Wildfires", "Floods"],
            "title": ["Fire", "Flood"],
            "latitude": [19.0, 200.0],
            "longitude": [72.0, 73.0],
            "event_timestamp": ["2024-01-01", "2024-01-02"],
            "status": ["open", "open"],
        })
        result = clean_eonet_events(df)
        assert len(result) == 1

    def test_clean_weather(self):
        df = pd.DataFrame({
            "forecast_timestamp": ["2024-01-01 00:00", "2024-01-01 01:00"],
            "temperature": [25.0, np.nan],
            "humidity": [60, 70],
            "wind_speed": [15.0, 20.0],
            "wind_direction": [180, 200],
            "precipitation": [0.0, 1.0],
            "pressure": [1013.0, 1012.0],
            "uv_index": [5.0, 4.0],
        })
        result = clean_weather(df)
        assert len(result) == 2
        assert result["temperature"].iloc[1] == 25.0  # filled with default

    def test_validate_coordinates(self):
        df = pd.DataFrame({
            "latitude": [19.0, 91.0, -100.0, 0.0],
            "longitude": [72.0, 73.0, 74.0, 200.0],
        })
        result = validate_coordinates(df)
        assert len(result) == 1


class TestScoring:
    def test_validate_risk_score(self):
        assert validate_risk_score(0)
        assert validate_risk_score(50)
        assert validate_risk_score(100)
        assert not validate_risk_score(-1)
        assert not validate_risk_score(101)
