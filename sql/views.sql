-- RiskSense AI — Curated Views

-- Current risk overview
CREATE OR REPLACE VIEW v_current_risk AS
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
FROM fact_environment_risk fer
JOIN dim_location dl ON fer.location_id = dl.location_id
WHERE fer.timestamp = (SELECT MAX(timestamp) FROM fact_environment_risk);

-- Hazard summary for dashboard
CREATE OR REPLACE VIEW v_active_hazards AS
SELECT
    event_id,
    category,
    title,
    latitude,
    longitude,
    event_timestamp
FROM raw_eonet_events
WHERE status = 'open';

-- Risk trend for forecasting
CREATE OR REPLACE VIEW v_risk_trend AS
SELECT
    fer.location_id,
    fer.timestamp,
    fer.risk_score,
    fer.risk_band,
    fer.predicted_risk_6h,
    fer.predicted_risk_12h
FROM fact_environment_risk fer
ORDER BY fer.timestamp;
