import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.database import get_connection, query

load_dotenv()

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/settings.yaml")
MODEL_DIR = Path("models")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [
        "hazard_count_10km", "hazard_count_50km", "hazard_count_100km",
        "nearest_hazard_distance_km", "wind_alignment_score",
        "temperature", "humidity", "wind_speed", "precipitation", "pressure", "uv_index",
        "hour", "day_of_week", "weekend",
        "temp_1h_change", "temp_3h_change", "wind_1h_change",
    ]
    existing = [c for c in feature_cols if c in df.columns]
    X = df[existing].copy()
    X = X.fillna(0)
    return X, existing


def train_target(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    config: dict,
) -> tuple[RandomForestRegressor, dict]:
    model_cfg = config["forecast"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=model_cfg["test_size"], random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "target": target_name,
        "mae": round(mean_absolute_error(y_test, y_pred), 3),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    logger.info("Model %s — MAE: %.3f, RMSE: %.3f", target_name, metrics["mae"], metrics["rmse"])
    return model, metrics


def run() -> dict:
    config = load_config()
    conn = get_connection()

    df = query(conn, """
        SELECT * FROM fact_environment_risk
        WHERE risk_score IS NOT NULL
        ORDER BY timestamp
    """)

    if df.empty or len(df) < 10:
        logger.warning("Not enough data to train model (%d rows)", len(df))
        if conn is not None:
            conn.close()
        return {"status": "insufficient_data", "rows": len(df)}

    X, feature_names = prepare_features(df)
    logger.info("Training with %d rows, %d features", len(X), len(feature_names))

    targets = config["forecast"]["targets"]
    models = {}
    all_metrics = {}

    risk_values = df["risk_score"].values
    for target in targets:
        horizon = 6 if target == "risk_6h" else 12
        y = pd.Series(
            [np.nan] * horizon + list(risk_values[:-horizon]) if len(risk_values) > horizon else risk_values
        )
        valid = y.notna() & (np.arange(len(y)) >= horizon)
        X_t = X[valid]
        y_t = y[valid]
        if len(X_t) < 10:
            logger.warning("Not enough data to train %s (%d rows)", target, len(X_t))
            continue
        model, metrics = train_target(X_t, y_t, target, config)
        models[target] = model
        all_metrics[target] = metrics

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "risk_forecast.joblib"
    payload = {"models": models, "feature_names": feature_names, "metrics": all_metrics}
    joblib.dump(payload, model_path)
    logger.info("Models saved to %s", model_path)

    if conn is not None:
        conn.close()
    return {"status": "success", "models": list(models.keys()), "metrics": all_metrics}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run()
    print(result)
