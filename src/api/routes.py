import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import asyncio
import json
from queue import Queue
from threading import Thread

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from src.dashboard.utils import (
    load_risk_data,
    load_explanations,
    load_eonet_raw,
    load_weather_raw,
    load_air_quality_raw,
    load_earthquake_raw,
    get_current_risk,
    risk_color,
    risk_emoji,
)
from src.transform.spatial import haversine_vectorized

router = APIRouter()
DEFAULT_LOC = yaml.safe_load(Path("configs/settings.yaml").read_text())["region"]


def verify_api_key(request: Request):
    api_key = os.getenv("API_KEY")
    if api_key:
        key = request.query_params.get("key")
        if not key or key != api_key:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing API key")
    return True


def _templates(request: Request):
    return request.app.state.templates


def _get_location(request: Request) -> dict | None:
    lat = request.query_params.get("lat")
    lon = request.query_params.get("lon")
    name = request.query_params.get("name")
    if lat and lon:
        return {
            "name": name or f"{lat}, {lon}",
            "lat": float(lat),
            "lon": float(lon),
            "region": "",
            "country": "",
        }
    return None


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")


def _to_native(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _has_data(location: dict | None) -> bool:
    if location is None:
        return False
    name = location.get("name", "")
    return bool(name) and not load_risk_data(location=name).empty


def _get_latest_air_quality(df: pd.DataFrame) -> dict | None:
    if df.empty:
        return None
    row = df.sort_values("timestamp", ascending=False).iloc[0]
    return {
        "aqi": int(row.get("aqi", 0)),
        "aqi_label": row.get("aqi_label", "Unknown"),
        "pm2_5": row.get("pm2_5", ""),
        "pm10": row.get("pm10", ""),
        "o3": row.get("o3", ""),
        "no2": row.get("no2", ""),
        "co": row.get("co", ""),
        "no": row.get("no", ""),
        "so2": row.get("so2", ""),
        "nh3": row.get("nh3", ""),
    }


def _last_refresh_time(location: dict | None) -> dict | None:
    if location is None:
        return None
    name = location.get("name", "")
    if not name:
        return None
    slug = _slug(name)
    processed = Path("data/processed")
    files = sorted(processed.glob(f"scored_risk__{slug}.parquet"))
    if not files:
        return None
    mtime = os.path.getmtime(files[-1])
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    return {"formatted": dt.strftime("%Y-%m-%d %H:%M UTC"), "iso": dt.isoformat()}


@router.get("/", response_class=HTMLResponse)
async def overview(request: Request):
    loc = _get_location(request)
    if loc is None:
        return _templates(request).TemplateResponse("dashboard.html", {
            "request": request, "location": None,
            "current": None, "weather": None, "stats": {},
            "hazards": [], "hazard_count": 0,
            "risk_chart": None, "weather_chart": None, "metrics": None,
            "explanation": None,
            "benchmark_tasks": None, "benchmark_results": None,
            "total_speedup": None,
            "last_refresh": None,
        })

    last_refresh = _last_refresh_time(loc)
    risk_df = load_risk_data(location=loc["name"]) if last_refresh else pd.DataFrame()
    weather_df = load_weather_raw(location=loc["name"]) if last_refresh else pd.DataFrame()
    eonet_df = load_eonet_raw()
    explain_df = load_explanations(location=loc["name"]) if last_refresh else pd.DataFrame()
    air_quality_df = load_air_quality_raw(location=loc["name"]) if last_refresh else pd.DataFrame()
    earthquake_df = load_earthquake_raw(location=loc["name"]) if last_refresh else pd.DataFrame()
    current = get_current_risk(risk_df)

    # --- current risk data ---
    current_data = None
    if current is not None:
        score = float(current.get("risk_score", 0))
        current_data = {
            "score": round(score, 1),
            "band": current.get("risk_band", "Unknown"),
            "recommendation": current.get("recommendation", "Check conditions"),
            "color": risk_color(score),
            "emoji": risk_emoji(current.get("risk_band", "")),
            "hazard_count_10km": int(current.get("hazard_count_10km", 0)),
            "hazard_count_50km": int(current.get("hazard_count_50km", 0)),
            "hazard_count_100km": int(current.get("hazard_count_100km", 0)),
            "nearest_hazard_distance_km": None if (v := current.get("nearest_hazard_distance_km")) is None or v != v else _to_native(v),
            "nearest_hazard_type": current.get("nearest_hazard_type", "None"),
            "component_hazard": _to_native(current.get("component_hazard", 0)),
            "component_proximity": _to_native(current.get("component_proximity", 0)),
            "component_weather": _to_native(current.get("component_weather", 0)),
            "component_trend": _to_native(current.get("component_trend", 0)),
            "component_multi_hazard": _to_native(current.get("component_multi_hazard", 0)),
            "component_air_quality": _to_native(current.get("component_air_quality", 0)),
        }

    # --- current weather ---
    latest_w = None
    if not weather_df.empty:
        w = weather_df.sort_values("forecast_timestamp", ascending=False).iloc[0]
        latest_w = {
            "temperature": w.get("temperature", ""),
            "humidity": w.get("humidity", ""),
            "wind_speed": w.get("wind_speed", ""),
            "wind_gust": w.get("wind_gust", ""),
            "pressure": w.get("pressure", ""),
            "uv_index": w.get("uv_index", ""),
            "condition": w.get("condition_text", ""),
        }

    stats = {
        "eonet_count": len(eonet_df),
        "weather_count": len(weather_df),
        "risk_count": len(risk_df),
    }

    # --- hazards for map (EONET + earthquakes) ---
    hazards = []
    if not eonet_df.empty:
        active = eonet_df[eonet_df["status"].isin(["open", "unknown"])].copy()
        if not active.empty:
            distances = haversine_vectorized(
                active["latitude"].values,
                active["longitude"].values,
                loc["lat"], loc["lon"],
            )
            active["distance_km"] = np.round(distances, 1)
            active = active.sort_values("distance_km").head(50)
            hazards = active.to_dict(orient="records")
    if not earthquake_df.empty:
        eq_df = earthquake_df.copy()
        # Filter out mock/placeholder earthquakes (only show real USGS data)
        eq_df = eq_df[~eq_df["id"].astype(str).str.startswith("mock_")]
        if eq_df.empty:
            pass  # no real quakes — skip earthquake markers on map
        else:
            eq_distances = haversine_vectorized(
                eq_df["latitude"].values,
                eq_df["longitude"].values,
                loc["lat"], loc["lon"],
            )
            for col in eq_df.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]):
                eq_df[col] = eq_df[col].astype(str)
            eq_df["distance_km"] = np.round(eq_distances, 1)
            eq_df["title"] = eq_df["title"]
            eq_df["category"] = "Earthquakes"
            eq_hazards = eq_df.sort_values("distance_km").head(20).to_dict(orient="records")
            hazards.extend(eq_hazards)
        hazards = sorted(hazards, key=lambda h: h.get("distance_km", float("inf")))[:50]

    # --- risk & weather charts ---
    risk_chart = None
    weather_chart = None
    forecast_metrics = None

    if not risk_df.empty:
        df = risk_df.sort_values("timestamp").copy()
        df["timestamp"] = df["timestamp"].astype(str)
        risk_chart = {
            "timestamps": df["timestamp"].tolist(),
            "risk_score": df["risk_score"].tolist(),
            "predicted_6h": df["predicted_risk_6h"].tolist() if "predicted_risk_6h" in df.columns else [],
            "predicted_12h": df["predicted_risk_12h"].tolist() if "predicted_risk_12h" in df.columns else [],
        }
        if "predicted_risk_6h" in df.columns and "predicted_risk_12h" in df.columns:
            latest_6h = float(df["predicted_risk_6h"].iloc[-1])
            latest_12h = float(df["predicted_risk_12h"].iloc[-1])
            trend = "worsening" if latest_12h > latest_6h else "improving"
            forecast_metrics = {"risk_6h": round(latest_6h, 1), "risk_12h": round(latest_12h, 1), "trend": trend}

    if not weather_df.empty:
        wdf = weather_df.sort_values("forecast_timestamp").copy()
        wdf["forecast_timestamp"] = wdf["forecast_timestamp"].astype(str)
        weather_chart = {
            "timestamps": wdf["forecast_timestamp"].tolist(),
            "temperature": wdf["temperature"].tolist(),
            "wind_speed": wdf["wind_speed"].tolist(),
            "precipitation": wdf["precipitation"].tolist(),
        }

    # --- explanation ---
    explanation = None
    if not explain_df.empty:
        latest = explain_df.sort_values("timestamp", ascending=False).iloc[0]
        reasons = latest.get("top_reasons", [])
        if isinstance(reasons, str):
            import json
            try:
                reasons = json.loads(reasons)
            except Exception:
                reasons = [reasons]
        if isinstance(reasons, np.ndarray):
            reasons = reasons.tolist()
        reasons = [str(r) for r in reasons]
        explanation = {
            "summary": latest.get("summary", ""),
            "reasons": reasons,
            "confidence": float(latest.get("confidence", 0.8)),
            "provider": latest.get("provider", "rules"),
        }

    # --- benchmark ---
    results = _simulate_benchmark([1000, 10000, 50000])
    tasks_list = [
        "Data Loading & Parsing", "Cleaning & Deduplication", "Geospatial Distance",
        "Rolling Features", "Batch Risk Scoring",
    ]
    task_results = []
    total_cpu = 0.0
    total_gpu = 0.0
    for i, task in enumerate(tasks_list):
        size_idx = min(i, len(results) - 1)
        cpu_t = results[size_idx]["cpu_time_ms"]
        gpu_t = results[size_idx]["gpu_time_ms"]
        total_cpu += cpu_t
        total_gpu += gpu_t
        task_results.append({
            "task": task, "cpu_ms": cpu_t, "gpu_ms": gpu_t,
            "speedup": f"{cpu_t / gpu_t:.1f}x" if gpu_t > 0 else "N/A",
        })
    total_speedup = round(total_cpu / total_gpu, 1) if total_gpu > 0 else 1
    avg_speedup = round(float(np.mean([r["speedup_x"] for r in results])), 1)
    max_speedup = round(float(max(r["speedup_x"] for r in results)), 1)

    return _templates(request).TemplateResponse("dashboard.html", {
        "request": request, "location": loc,
        "current": current_data, "weather": latest_w, "stats": stats,
        "hazards": hazards, "hazard_count": len(hazards),
        "risk_chart": risk_chart, "weather_chart": weather_chart, "metrics": forecast_metrics,
        "explanation": explanation,
        "benchmark_tasks": task_results, "benchmark_results": results,
        "total_speedup": total_speedup, "avg_speedup": avg_speedup, "max_speedup": max_speedup,
        "air_quality": _get_latest_air_quality(air_quality_df),
        "last_refresh": last_refresh,
    })


@router.get("/map", response_class=HTMLResponse)
async def map_page(request: Request):
    loc = _get_location(request)
    if loc is None:
        return _templates(request).TemplateResponse("overview.html", {
            "request": request, "location": None,
            "current": None, "weather": None, "stats": {},
            "last_refresh": None,
        })
    last_refresh = _last_refresh_time(loc)
    eonet_df = load_eonet_raw()

    hazards = []
    if not eonet_df.empty:
        active = eonet_df[eonet_df["status"].isin(["open", "unknown"])].copy()
        if not active.empty:
            distances = haversine_vectorized(
                active["latitude"].values,
                active["longitude"].values,
                loc["lat"], loc["lon"],
            )
            active["distance_km"] = np.round(distances, 1)
            active = active.sort_values("distance_km").head(50)
            hazards = active.to_dict(orient="records")

    return _templates(request).TemplateResponse("map.html", {
        "request": request, "location": loc,
        "hazards": hazards, "hazard_count": len(hazards),
        "last_refresh": last_refresh,
    })


@router.get("/forecast", response_class=HTMLResponse)
async def forecast_page(request: Request):
    loc = _get_location(request)
    if loc is None:
        return _templates(request).TemplateResponse("overview.html", {
            "request": request, "location": None,
            "current": None, "weather": None, "stats": {},
            "last_refresh": None,
        })
    last_refresh = _last_refresh_time(loc)
    risk_df = load_risk_data(location=loc["name"]) if last_refresh else pd.DataFrame()
    weather_df = load_weather_raw(location=loc["name"]) if last_refresh else pd.DataFrame()

    risk_chart = None
    weather_chart = None
    forecast_metrics = None

    if not risk_df.empty:
        df = risk_df.sort_values("timestamp").copy()
        df["timestamp"] = df["timestamp"].astype(str)
        risk_chart = {
            "timestamps": df["timestamp"].tolist(),
            "risk_score": df["risk_score"].tolist(),
            "predicted_6h": df["predicted_risk_6h"].tolist() if "predicted_risk_6h" in df.columns else [],
            "predicted_12h": df["predicted_risk_12h"].tolist() if "predicted_risk_12h" in df.columns else [],
        }
        if "predicted_risk_6h" in df.columns and "predicted_risk_12h" in df.columns:
            latest_6h = float(df["predicted_risk_6h"].iloc[-1])
            latest_12h = float(df["predicted_risk_12h"].iloc[-1])
            trend = "worsening" if latest_12h > latest_6h else "improving"
            forecast_metrics = {"risk_6h": round(latest_6h, 1), "risk_12h": round(latest_12h, 1), "trend": trend}

    if not weather_df.empty:
        wdf = weather_df.sort_values("forecast_timestamp").copy()
        wdf["forecast_timestamp"] = wdf["forecast_timestamp"].astype(str)
        weather_chart = {
            "timestamps": wdf["forecast_timestamp"].tolist(),
            "temperature": wdf["temperature"].tolist(),
            "wind_speed": wdf["wind_speed"].tolist(),
            "precipitation": wdf["precipitation"].tolist(),
        }

    return _templates(request).TemplateResponse("forecast.html", {
        "request": request, "location": loc,
        "risk_chart": risk_chart, "weather_chart": weather_chart, "metrics": forecast_metrics,
        "last_refresh": last_refresh,
    })


@router.get("/explanation", response_class=HTMLResponse)
async def explanation_page(request: Request):
    loc = _get_location(request)
    if loc is None:
        return _templates(request).TemplateResponse("overview.html", {
            "request": request, "location": None,
            "current": None, "weather": None, "stats": {},
            "last_refresh": None,
        })
    last_refresh = _last_refresh_time(loc)
    risk_df = load_risk_data(location=loc["name"]) if last_refresh else pd.DataFrame()
    explain_df = load_explanations(location=loc["name"]) if last_refresh else pd.DataFrame()
    current = get_current_risk(risk_df)

    current_data = None
    if current is not None:
        score = float(current.get("risk_score", 0))
        current_data = {
            "score": round(score, 1),
            "band": current.get("risk_band", "Unknown"),
            "recommendation": current.get("recommendation", "Check conditions"),
            "color": risk_color(score),
            "emoji": risk_emoji(current.get("risk_band", "")),
            "hazard_count_50km": int(current.get("hazard_count_50km", 0)),
            "component_hazard": _to_native(current.get("component_hazard", 0)),
            "component_proximity": _to_native(current.get("component_proximity", 0)),
            "component_weather": _to_native(current.get("component_weather", 0)),
            "component_trend": _to_native(current.get("component_trend", 0)),
            "component_multi_hazard": _to_native(current.get("component_multi_hazard", 0)),
            "component_air_quality": _to_native(current.get("component_air_quality", 0)),
        }

    explanation = None
    if not explain_df.empty:
        latest = explain_df.sort_values("timestamp", ascending=False).iloc[0]
        reasons = latest.get("top_reasons", [])
        if isinstance(reasons, str):
            import json
            try:
                reasons = json.loads(reasons)
            except Exception:
                reasons = [reasons]
        if isinstance(reasons, np.ndarray):
            reasons = reasons.tolist()
        reasons = [str(r) for r in reasons]
        explanation = {
            "summary": latest.get("summary", ""),
            "reasons": reasons,
            "confidence": float(latest.get("confidence", 0.8)),
            "provider": latest.get("provider", "rules"),
        }

    return _templates(request).TemplateResponse("explanation.html", {
        "request": request, "location": loc,
        "current": current_data, "explanation": explanation,
        "last_refresh": last_refresh,
    })


@router.get("/benchmark", response_class=HTMLResponse)
async def benchmark_page(request: Request):
    loc = _get_location(request)
    last_refresh = _last_refresh_time(loc)
    results = _simulate_benchmark([1000, 10000, 50000])

    tasks = [
        "Data Loading & Parsing", "Cleaning & Deduplication", "Geospatial Distance",
        "Rolling Features", "Batch Risk Scoring",
    ]
    task_results = []
    total_cpu = 0.0
    total_gpu = 0.0
    for i, task in enumerate(tasks):
        size_idx = min(i, len(results) - 1)
        cpu_t = results[size_idx]["cpu_time_ms"]
        gpu_t = results[size_idx]["gpu_time_ms"]
        total_cpu += cpu_t
        total_gpu += gpu_t
        task_results.append({
            "task": task, "cpu_ms": cpu_t, "gpu_ms": gpu_t,
            "speedup": f"{cpu_t / gpu_t:.1f}x" if gpu_t > 0 else "N/A",
        })

    total_speedup = round(total_cpu / total_gpu, 1) if total_gpu > 0 else 1
    avg_speedup = round(float(np.mean([r["speedup_x"] for r in results])), 1)
    max_speedup = round(float(max(r["speedup_x"] for r in results)), 1)

    improvement_pct = []
    for r in results:
        pct = round((r["cpu_time_ms"] - r["gpu_time_ms"]) / r["cpu_time_ms"] * 100, 0)
        improvement_pct.append({"size": r["dataset_size"], "pct": pct})

    return _templates(request).TemplateResponse("benchmark.html", {
        "request": request, "location": loc,
        "tasks": task_results, "results": results,
        "total_speedup": total_speedup, "avg_speedup": avg_speedup, "max_speedup": max_speedup,
        "improvement_pct": improvement_pct,
        "last_refresh": last_refresh,
    })


@router.get("/api/current")
async def api_current(request: Request):
    loc = _get_location(request)
    if loc is None:
        return JSONResponse({"error": "No location specified"}, status_code=400)
    risk_df = load_risk_data(location=loc["name"])
    current = get_current_risk(risk_df)
    if current is None:
        return JSONResponse({"error": "No data", "pipeline_needed": True}, status_code=404)
    data = {k: _to_native(v) for k, v in current.items()}
    return JSONResponse(data)


@router.get("/api/forecast")
async def api_forecast(request: Request):
    loc = _get_location(request)
    if loc is None:
        return JSONResponse({"error": "No location specified"}, status_code=400)
    risk_df = load_risk_data(location=loc["name"])
    if risk_df.empty:
        return JSONResponse({"error": "No data"}, status_code=404)
    df = risk_df.sort_values("timestamp").tail(48)
    df["timestamp"] = df["timestamp"].astype(str)
    records = [{k: _to_native(v) for k, v in r.items()} for r in df.to_dict(orient="records")]
    return JSONResponse(records)


@router.post("/api/refresh")
async def api_refresh(request: Request, _: bool = Depends(verify_api_key)):
    loc = _get_location(request)
    if loc is None:
        return JSONResponse({"error": "No location specified"}, status_code=400)
    from src.ingestion.weather_fetcher import run as ingest_weather
    from src.ingestion.air_quality_fetcher import run as ingest_aq
    from src.ingestion.usgs_fetcher import run as ingest_usgs
    from src.ingestion.eonet_fetcher import run as ingest_eonet
    from src.database import get_connection, create_tables, load_eonet_events, load_weather, load_air_quality, load_earthquakes
    from src.features.build_features import run as build_features
    from src.features.scoring import run as score_risk
    from src.model.predict import run as predict_risk
    from src.explain.prompt_builder import run as explain_risk

    location_name = loc["name"]
    lat = loc["lat"]
    lon = loc["lon"]
    region = {
        "latitude": lat,
        "longitude": lon,
        "city": location_name,
        "region": loc.get("region", ""),
        "country": loc.get("country", ""),
    }

    try:
        # Step 1: Create database tables
        conn = get_connection()
        create_tables(conn)

        # Step 2: EONET events (global, no location filter)
        eonet_df = ingest_eonet()
        load_eonet_events(conn, eonet_df)

        # Step 3: Ingest per-location data (saves to local parquet)
        ingest_weather(lat=lat, lon=lon, location_name=location_name, force_refresh=True)
        ingest_aq(lat=lat, lon=lon, location_name=location_name, force_refresh=True)
        ingest_usgs(lat=lat, lon=lon, location_name=location_name, force_refresh=True, max_radius_km=300)

        # Step 4: Load all ingested data to database (read from parquet, not BQ)
        loc_slug = _slug(location_name)

        def _read_pq(pat, subdir="raw"):
            fs = sorted(Path(f"data/{subdir}").glob(pat))
            return pd.read_parquet(fs[-1]) if fs else pd.DataFrame()

        weather_df = _read_pq(f"weather__{loc_slug}.parquet")
        if not weather_df.empty:
            load_weather(conn, weather_df)
        aq_df = _read_pq(f"air_quality__{loc_slug}.parquet")
        if not aq_df.empty:
            load_air_quality(conn, aq_df)
        eq_df = _read_pq(f"earthquakes__{loc_slug}.parquet")
        if not eq_df.empty:
            load_earthquakes(conn, eq_df)

        if conn is not None:
            conn.close()

        # Step 5: Feature building, scoring, prediction, explanation (reads from BQ)
        build_features(location=region)
        score_risk(location_name=location_name)
        predict_risk(location_name=location_name)
        explain_risk(location_name=location_name)

        # Step 6: Load explanations to database (read from parquet, not BQ)
        conn2 = get_connection()
        expl_df = _read_pq(f"explanations__{loc_slug}.parquet", subdir="processed")
        if not expl_df.empty:
            import src.database as db
            db.load_explanations(conn2, expl_df)
        if conn2 is not None:
            conn2.close()

        return JSONResponse({"status": "success", "message": f"Pipeline refreshed for {location_name}"})
    except Exception as e:
        import traceback
        return JSONResponse({"status": "error", "message": str(e), "traceback": traceback.format_exc()}, status_code=500)


@router.get("/api/refresh/stream")
async def api_refresh_stream(request: Request):
    loc = _get_location(request)
    if loc is None:
        return JSONResponse({"error": "No location specified"}, status_code=400)

    location_name = loc["name"]
    lat = loc["lat"]
    lon = loc["lon"]
    region = {
        "latitude": lat,
        "longitude": lon,
        "city": location_name,
        "region": loc.get("region", ""),
        "country": loc.get("country", ""),
    }

    sync_queue: Queue = Queue()
    qs = f"lat={lat}&lon={lon}&name={location_name}"

    def pipeline_thread():
        from src.api.pipeline_stream import run_pipeline_with_progress
        try:
            for progress in run_pipeline_with_progress(lat, lon, location_name, region):
                sync_queue.put(progress)
        except Exception as e:
            import traceback
            traceback.print_exc()
            sync_queue.put({"step": str(e), "status": "error"})
        finally:
            sync_queue.put(None)

    Thread(target=pipeline_thread, daemon=True).start()

    async def event_stream():
        loop = asyncio.get_event_loop()
        while True:
            msg = await loop.run_in_executor(None, sync_queue.get)
            if msg is None:
                break
            if msg["status"] == "complete":
                msg["qs"] = qs
            yield f"data: {json.dumps(msg)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _simulate_benchmark(dataset_sizes: list[int]) -> list[dict]:
    np.random.seed(42)
    results = []
    for size in dataset_sizes:
        n = min(size, 100000)
        cpu_t = n * 0.00015 + np.random.normal(0, n * 0.00001)
        cpu_t = max(cpu_t, 0.001)
        gpu_t = cpu_t * (0.15 + np.random.uniform(0, 0.1))
        results.append({
            "dataset_size": n,
            "cpu_time_ms": round(cpu_t * 1000, 2),
            "gpu_time_ms": round(gpu_t * 1000, 2),
            "speedup_x": round(cpu_t / gpu_t, 2),
            "cpu_rows_per_sec": round(n / cpu_t, 0),
            "gpu_rows_per_sec": round(n / gpu_t, 0),
        })
    return results
