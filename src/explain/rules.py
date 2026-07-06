import logging

logger = logging.getLogger(__name__)


def build_explanation(row: dict) -> dict:
    score = row.get("risk_score", 0)
    band = row.get("risk_band", "Unknown")
    recommendation = row.get("recommendation", "Check conditions")
    hazard_count = row.get("hazard_count_100km", 0)
    nearest_dist = row.get("nearest_hazard_distance_km")
    nearest_type = row.get("nearest_hazard_type", "None")
    hazard_types = row.get("hazard_types", "")
    wind_speed = row.get("wind_speed", 0) or 0
    temp_change = row.get("temp_1h_change", 0) or 0

    reasons = []

    if hazard_count > 0 and nearest_type != "None":
        if nearest_dist is not None and nearest_dist <= 10:
            reasons.append(f"A {nearest_type} is within {nearest_dist:.0f} km of your location")
        elif nearest_dist is not None and nearest_dist <= 50:
            reasons.append(f"A {nearest_type} is {nearest_dist:.0f} km away — moderately close")
        elif nearest_dist is not None:
            reasons.append(f"A {nearest_type} is reported {nearest_dist:.0f} km away")

    if hazard_count > 1:
        reasons.append(f"There are {hazard_count} active hazards in the broader area")

    if wind_speed >= 60:
        reasons.append(f"Wind speed is {wind_speed:.0f} km/h — extreme winds amplify risk")
    elif wind_speed >= 30:
        reasons.append(f"Wind speed is {wind_speed:.0f} km/h — moderate winds may carry hazard effects")

    if temp_change is not None and temp_change >= 3:
        reasons.append("Temperature is rising sharply, indicating worsening conditions")
    elif temp_change is not None and temp_change <= -3:
        reasons.append("Temperature is dropping, which may indicate improving conditions")

    if score <= 39:
        summary = "Conditions appear safe for outdoor activity."
        if reasons:
            summary = "Conditions are generally safe, but note: " + reasons[0].lower() + "."
    elif score <= 69:
        summary = "Caution is advised for outdoor activity."
        if reasons:
            summary = "Caution advised: " + reasons[0].lower()
            if len(reasons) > 1:
                summary += f" Additionally, {reasons[1].lower()}."
    else:
        summary = "Avoid outdoor activity today."
        if reasons:
            summary = f"Avoid outdoor activity today. {reasons[0].capitalize()}"
            if len(reasons) > 1:
                summary += f" {reasons[1].capitalize()}"
            summary += ". Conditions are likely to worsen before they improve."

    return {
        "recommendation": recommendation,
        "risk_band": band,
        "risk_score": round(score, 1),
        "summary": summary,
        "top_reasons": reasons[:3],
        "confidence": 0.85 if hazard_count > 0 else 0.7,
    }


def run(row: dict) -> dict:
    return build_explanation(row)
