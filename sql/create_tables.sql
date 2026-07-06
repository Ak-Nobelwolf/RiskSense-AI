-- RiskSense AI — DuckDB Table Schemas (lenient, MVP-friendly)
-- These mirror the BigQuery schema for future migration.

DROP TABLE IF EXISTS fact_environment_risk;
DROP TABLE IF EXISTS dim_calendar;
DROP TABLE IF EXISTS dim_location;
DROP TABLE IF EXISTS raw_weather;
DROP TABLE IF EXISTS raw_eonet_events;

CREATE TABLE raw_eonet_events (
    event_id VARCHAR,
    category VARCHAR,
    title VARCHAR,
    status VARCHAR,
    event_timestamp TIMESTAMP,
    latitude DOUBLE,
    longitude DOUBLE
);

CREATE TABLE raw_weather (
    forecast_timestamp TIMESTAMP,
    temperature DOUBLE,
    humidity DOUBLE,
    wind_speed DOUBLE,
    wind_gust DOUBLE,
    wind_direction DOUBLE,
    precipitation DOUBLE,
    pressure DOUBLE,
    uv_index DOUBLE,
    condition_text VARCHAR,
    location_id INTEGER DEFAULT 1,
    source_name VARCHAR DEFAULT 'openweathermap',
    ingestion_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE dim_location (
    location_id INTEGER,
    location_name VARCHAR,
    latitude DOUBLE,
    longitude DOUBLE,
    city VARCHAR,
    region VARCHAR,
    country VARCHAR
);

CREATE TABLE fact_environment_risk (
    location_id INTEGER DEFAULT 1,
    timestamp TIMESTAMP,
    hazard_count_10km INTEGER DEFAULT 0,
    hazard_count_50km INTEGER DEFAULT 0,
    hazard_count_100km INTEGER DEFAULT 0,
    nearest_hazard_distance_km DOUBLE,
    nearest_hazard_type VARCHAR,
    hazard_types VARCHAR,
    hazard_severity_score DOUBLE DEFAULT 0,
    wind_alignment_score DOUBLE DEFAULT 0,
    trend_score DOUBLE DEFAULT 0,
    temperature DOUBLE,
    humidity DOUBLE,
    wind_speed DOUBLE,
    wind_direction DOUBLE,
    precipitation DOUBLE,
    pressure DOUBLE,
    uv_index DOUBLE,
    aqi INTEGER,
    co DOUBLE,
    so2 DOUBLE,
    max_eq_magnitude DOUBLE DEFAULT 0,
    hour INTEGER,
    day_of_week INTEGER,
    weekend INTEGER,
    temp_3h_avg DOUBLE,
    temp_6h_avg DOUBLE,
    temp_1h_change DOUBLE,
    temp_3h_change DOUBLE,
    wind_3h_avg DOUBLE,
    wind_1h_change DOUBLE,
    precip_3h_sum DOUBLE,
    risk_score DOUBLE,
    risk_band VARCHAR,
    recommendation VARCHAR,
    component_hazard DOUBLE DEFAULT 0,
    component_proximity DOUBLE DEFAULT 0,
    component_weather DOUBLE DEFAULT 0,
    component_trend DOUBLE DEFAULT 0,
    component_multi_hazard DOUBLE DEFAULT 0,
    component_air_quality DOUBLE DEFAULT 0,
    predicted_risk_6h DOUBLE,
    predicted_risk_12h DOUBLE,
    confidence_score DOUBLE DEFAULT 0.8,
    explanation_text VARCHAR
);

-- Insert default Mumbai location
INSERT INTO dim_location VALUES (1, 'Mumbai Central', 19.0760, 72.8777, 'Mumbai', 'Maharashtra', 'India');
