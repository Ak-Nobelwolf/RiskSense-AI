import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def _build_prompt(row: dict) -> str:
    return f"""
You are an environmental safety advisor. Given the following data, provide a recommendation.

Location: {row.get('location_name', 'Mumbai')}
Current risk score: {row.get('risk_score', 0)}/100
Risk band: {row.get('risk_band', 'Unknown')}
Active hazards within 100 km: {row.get('hazard_types', 'None')}
Nearest hazard: {row.get('nearest_hazard_type', 'None')} at {row.get('nearest_hazard_distance_km', 'N/A')} km
Current weather: {row.get('temperature', 'N/A')}°C, wind {row.get('wind_speed', 'N/A')} km/h, humidity {row.get('humidity', 'N/A')}%
6h forecast risk: {row.get('predicted_risk_6h', 'N/A')}
12h forecast risk: {row.get('predicted_risk_12h', 'N/A')}

Provide:
1. Recommendation label (one of: Go now, Safe with caution, Wait until later, Avoid today)
2. Confidence score (0-1)
3. One short paragraph explaining the reasoning
4. Top 3 contributing factors

Respond in JSON format with keys: recommendation, confidence, explanation, top_reasons.
""".strip()


def explain_with_gemini(row: dict) -> dict | None:
    if not GEMINI_API_KEY:
        logger.debug("No GEMINI_API_KEY set, skipping Gemini call")
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = _build_prompt(row)
        response = model.generate_content(prompt)
        text = response.text.strip()
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(text)
        logger.info("Gemini explanation generated successfully")
        return result
    except ImportError:
        logger.warning("google-generativeai not installed. Run: pip install google-generativeai")
        return None
    except Exception as e:
        logger.warning("Gemini API call failed: %s", e)
        return None
