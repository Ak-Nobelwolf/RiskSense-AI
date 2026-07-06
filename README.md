# RiskSense AI — Outdoor Safety Decision Support

**Should I go outside now, later, or not today?**

RiskSense AI combines NASA EONET hazard data, OpenWeatherMap forecasts, USGS earthquakes, and air quality into a single 0–100 risk score with forward-looking ML predictions and actionable recommendations. Deployable locally or on Google Cloud free tier.

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

## Setup

### Prerequisites

- Python 3.10+
- (Optional) GCP free tier account for cloud deployment

### Installation

```bash
git clone <repo-url>
cd RiskSense AI
python -m venv .venv
.venv\Scripts\activate         # Windows
source .venv/bin/activate      # Linux/Mac
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in the fields you need. At minimum:

```env
# Local dev only: leave empty for DuckDB mode
GOOGLE_APPLICATION_CREDENTIALS=
GCP_PROJECT_ID=
GCS_BUCKET_NAME=
BIGQUERY_DATASET=risksense

# Without this, weather uses mock data
OPENWEATHERMAP_API_KEY=

# Optional: AI explanations vs rule-based fallback
GEMINI_API_KEY=
```

## Usage (Local)

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8503
```

Open http://localhost:8503 in your browser.

**First visit:** Tables are created automatically on startup. Hit the refresh endpoint to hydrate data:

```bash
curl -X POST "http://localhost:8503/api/refresh?lat=19.076&lon=72.8777&name=Mumbai"
```

Then reload the dashboard.

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

BigQuery adds network latency per query, but scales to much larger datasets and requires zero local infrastructure.

## Cloud Deployment

```bash
gcloud run deploy risksense-ai --source . --region=asia-south1 \
  --set-env-vars=GCP_PROJECT_ID=risksense-ai,GCS_BUCKET_NAME=risksense-data,BIGQUERY_DATASET=risksense,OPENWEATHERMAP_API_KEY=your_key,GEMINI_API_KEY=your_key \
  --allow-unauthenticated
```

Deploys to Cloud Run (free tier: 2M requests/month). Tables created on first request. Data persisted in BigQuery + GCS across container restarts.

## Testing

```bash
pytest tests/ -v
```

## Limitations

- **Batch refresh**: Not real-time; manual or cron-triggered pipeline
- **Simple forecast**: Random Forest baseline; no deep learning
- **Mock USGS data**: Behind corporate proxy (falls back to generated data)
- **API dependency**: EONET, OpenWeatherMap availability affects pipeline
- **No user auth**: Single-user dashboard

## Roadmap

- [x] Phase 1: Skeleton + schemas
- [x] Phase 2: Data ingestion (EONET, OWM, USGS, AQ)
- [x] Phase 3: Transformation + features
- [x] Phase 4: Risk scoring + ML forecast
- [x] Phase 5: Explanation engine
- [x] Phase 6: FastAPI dashboard + SSE pipeline
- [x] Phase 7: Cloud deployment (GCS + BigQuery + Cloud Run)
- [ ] Real-time streaming
- [ ] Mobile notifications
- [ ] Historical trend analysis

## License

MIT
