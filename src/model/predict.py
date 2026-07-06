import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.database import get_connection, query, load_feature_table
from src.ingestion.storage_writer import store_parquet

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "risk_forecast.joblib"


def load_model():
    if not MODEL_PATH.exists():
        logger.warning("No trained model found at %s", MODEL_PATH)
        return None
    return joblib.load(MODEL_PATH)


def prepare_prediction_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    available = [c for c in feature_names if c in df.columns]
    X = df[available].copy()
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names].fillna(0)
    return X


def predict_risk(
    df: pd.DataFrame,
    payload: dict,
) -> pd.DataFrame:
    models = payload["models"]
    feature_names = payload["feature_names"]

    X = prepare_prediction_features(df, feature_names)
    result_df = df.copy()

    for target, model in models.items():
        preds = model.predict(X)
        result_df[target] = np.round(preds, 1)

    return result_df


def run(feature_df: pd.DataFrame | None = None, location_name: str | None = None) -> pd.DataFrame:
    payload = load_model()
    if payload is None or not payload["models"]:
        logger.warning("No trained models to predict with")
        return pd.DataFrame()

    conn = get_connection()
    if feature_df is None:
        feature_df = query(conn, "SELECT * FROM fact_environment_risk")

    if feature_df.empty:
        logger.warning("No feature data for prediction")
        if conn is not None:
            conn.close()
        return feature_df

    result_df = predict_risk(feature_df, payload)
    for target in payload["models"]:
        if target in result_df.columns:
            col_name = f"predicted_{target}"
            result_df[col_name] = result_df[target]
            result_df = result_df.drop(columns=[target])

    slug = location_name.lower().replace(" ", "_").replace(",", "") if location_name else None
    path = store_parquet(result_df, "scored_risk.parquet", subdir="processed", location_slug=slug)
    load_feature_table(conn, result_df, "fact_environment_risk")

    logger.info("Predicted risk for %d rows for '%s', saved to %s", len(result_df), location_name or "default", path)
    if conn is not None:
        conn.close()
    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
