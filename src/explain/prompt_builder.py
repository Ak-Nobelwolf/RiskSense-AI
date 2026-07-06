import json
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.database import get_connection, query
from src.explain.rules import build_explanation
from src.explain.gemini_client import explain_with_gemini
from src.ingestion.storage_writer import store_parquet
from src.dashboard.utils import load_risk_data

load_dotenv()

logger = logging.getLogger(__name__)


def run(location_name: str = "mumbai") -> pd.DataFrame:
    risk_df = load_risk_data(location=location_name)

    if risk_df.empty:
        logger.warning("No risk data for '%s' — explanation generation", location_name)
        return pd.DataFrame()

    explanations = []
    for _, row in risk_df.iterrows():
        row_dict = row.to_dict()
        row_dict["location_name"] = location_name

        provider = "rules"
        try:
            result = explain_with_gemini(row_dict)
            if result:
                provider = "gemini"
        except Exception:
            result = None

        if not result:
            result = build_explanation(row_dict)

        result["timestamp"] = row_dict.get("timestamp")
        result["provider"] = provider
        explanations.append(result)

    expl_df = pd.DataFrame(explanations)
    slug = location_name.lower().replace(" ", "_").replace(",", "")
    path = store_parquet(expl_df, "explanations.parquet", subdir="processed", location_slug=slug)
    logger.info("Generated %d explanations for '%s', saved to %s", len(expl_df), location_name, path)
    return expl_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
