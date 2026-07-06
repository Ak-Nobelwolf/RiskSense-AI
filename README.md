# RiskSense AI — Outdoor Safety Decision Support

**Should I go outside now, later, or not today?**

RiskSense AI is a deployed outdoor safety decision-support tool that combines hazard, weather, air-quality, and earthquake data into a single 0-100 risk score with short-term forecast output and plain-English guidance. It helps users make faster decisions about whether conditions are safe enough to go outside.

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| **NASA EONET** | Active hazards (wildfires, floods, storms, volcanoes) | Free API, no key |
| **OpenWeatherMap** | Weather 3h forecast + air pollution (PM2.5, PM10, O₃, CO, NO₂, SO₂) | Free tier, API key |
| **USGS** | Real-time earthquakes worldwide | Free API, no key |
| **Nominatim (OSM)** | Geocoding — any searchable location | Free API, no key |

## Architecture

```
EONET ─┐
OWM    ├── ingestion → validate → feature engineering → risk scoring → forecast → explanation → FastAPI dashboard
USGS ──┘
         ↓                          ↓
   Cloud Storage (Parquet)    BigQuery (serverless SQL)
```

- **Ingestion**: Fetch → normalize → store Parquet + load to BigQuery
- **Storage**: Local Parquet + GCS backup (cloud) or DuckDB (local dev)
- **Scoring**: Explainable rule-based formula (0–100, 6 components)
- **Forecast**: Random Forest predicting 6h and 12h risk
- **Explanation**: Rule-based (default) or Gemini API (optional)
- **Dashboard**: FastAPI + Jinja2 + Plotly + Leaflet + Tailwind
- **Deployment**: Local (uvicorn) or Cloud Run (serverless, free tier)

## Project Structure

```
project-root/
├── configs/settings.yaml      # Tunable parameters
├── .env.example               # Environment variable template
├── Dockerfile                 # Cloud Run container
├── requirements.txt           # Python dependencies
├── src/
│   ├── api/                   # FastAPI server + templates + static
│   │   ├── routes.py          # All HTTP endpoints
│   │   ├── app.py             # FastAPI app factory
│   │   ├── pipeline_stream.py # SSE pipeline progress
│   │   ├── templates/         # Jinja2 HTML
│   │   └── static/            # CSS, JS, images
│   ├── ingestion/             # API fetchers + GCS client
│   │   ├── eonet_fetcher.py   # NASA EONET
│   │   ├── weather_fetcher.py # OpenWeatherMap
│   │   ├── air_quality_fetcher.py
│   │   ├── usgs_fetcher.py    # Earthquakes
│   │   ├── gcs_client.py      # Google Cloud Storage
│   │   └── storage_writer.py  # Local + GCS parquet I/O
│   ├── transform/             # Clean, validate, spatial
│   ├── features/              # Feature engineering + scoring
│   ├── model/                 # Train + predict (6h/12h)
│   ├── explain/               # Rule-based + Gemini explainer
│   ├── database.py            # DuckDB/BigQuery dispatch
│   ├── database_bq.py         # BigQuery DDL + loads + queries
│   └── dashboard/             # Dashboard data loaders
├── data/                      # Local parquet cache (gitignored)
└── sql/                       # DuckDB schemas (local dev only)
```

## What It Does

- Lets a user search any location and see a current safety assessment
- Ingests live hazard, weather, air-quality, and earthquake data
- Builds features, scores risk, and generates 6h/12h forecast output
- Renders an interactive dashboard with a hazard map, charts, and a risk summary
- Uses Gemini API for AI-generated explanations when available, with a rule-based fallback

## Built With

- **FastAPI** for the web app and API layer
- **Jinja2** for server-rendered templates
- **Plotly** for time-series charts
- **Leaflet** for the interactive hazard map
- **Tailwind CSS** for the UI styling
- **Pandas / NumPy** for data processing
- **scikit-learn** for the forecast model
- **DuckDB** for local analytics support
- **BigQuery** and **Cloud Storage** for Google Cloud storage and analytics support
- **NASA EONET**, **OpenWeatherMap**, **USGS**, and **Nominatim** as data sources
- **Gemini API** for AI-generated explanations

### Pipeline Steps

The `/api/refresh` endpoint runs the full pipeline in order:
1. Create BigQuery/DuckDB tables (idempotent)
2. Ingest EONET events → load to database
3. Ingest weather, air quality, earthquakes → save Parquet
4. Load all ingested data to database
5. Build features → score risk → predict 6h/12h → generate explanations

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard overview (risk, weather, hazards, charts) |
| `GET /map` | Full-screen hazard map |
| `GET /forecast` | Risk forecast charts |
| `GET /explanation` | AI/rule-based explanation |
| `GET /benchmark` | CPU vs BigQuery speed comparison |
| `GET /api/current` | Current risk score JSON |
| `GET /api/forecast` | Forecast data JSON |
| `POST /api/refresh` | Trigger full pipeline for a location |
| `GET /api/refresh/stream` | SSE streaming pipeline progress |

## Benchmark

RiskSense AI can run on local DuckDB (CPU) or Cloud BigQuery:

| Task | DuckDB | BigQuery | Speedup |
|------|--------|----------|---------|
| Data loading | ~0.3s | ~1.2s | 0.25× |
| Query (100 rows) | ~0.001s | ~0.4s | 0.002× |
| Full pipeline (incl. API calls) | ~25s | ~30s | 0.83× |

These benchmark values are simulated in the app UI to illustrate the relative tradeoff between local DuckDB and cloud BigQuery paths. BigQuery adds network latency per query, but scales to much larger datasets and requires zero local infrastructure.

## Cloud Deployment

Deploys to Cloud Run (free tier: 2M requests/month). Tables created on first request. Data persisted in BigQuery + GCS across container restarts.

## Testing

```bash
pytest tests/ -v
```

## Limitations

- **Batch refresh**: Not real-time; manual or cron-triggered pipeline
- **Simple forecast**: Random Forest baseline; no deep learning
- **API dependency**: EONET, OpenWeatherMap availability affects pipeline
- **No user auth**: Single-user dashboard
