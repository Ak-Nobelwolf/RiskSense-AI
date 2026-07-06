# 🌍 RiskSense AI – AI Powered Outdoor Safety Decision Support
![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google%20Cloud-4285F4?logo=googlecloud&logoColor=white)
![BigQuery](https://img.shields.io/badge/BigQuery-4285F4?logo=googlebigquery&logoColor=white)
![Cloud Run](https://img.shields.io/badge/Cloud%20Run-4285F4?logo=googlecloud&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-AI-blue)
![NVIDIA](https://img.shields.io/badge/NVIDIA-RAPIDS-76B900?logo=nvidia&logoColor=white)
<p align="center">

**Should I go outside now, later, or not today?**

*An AI-powered environmental risk intelligence platform built on Google Cloud with NVIDIA acceleration.*

</p>

---

## 🚀 Live Demo

🌐 **Application**

https://risksense-ai-789818308989.asia-south1.run.app/

---

# 📖 About the Project

Every day, millions of people make outdoor decisions based on incomplete information.

They check:

* 🌦 Weather apps
* 🌫 Air Quality apps
* 🔥 Wildfire alerts
* 🌍 Earthquake reports
* 📰 Local news

RiskSense AI combines all these sources into **one intelligent decision-support platform** that answers a single question:

> **"Is it safe to go outside?"**

The platform collects live environmental data, processes it through an explainable risk engine, predicts future conditions using Machine Learning, and presents everything in an interactive dashboard with a unified **0–100 Environmental Risk Score**.

#### The application is fully deployed on Google Cloud Run and is publicly accessible through the live demo.
---

# 🎯 Problem Statement

People currently rely on multiple disconnected applications to determine outdoor safety.

RiskSense AI eliminates this fragmentation by combining:

* Weather
* Air Quality
* Wildfires
* Volcanoes
* Earthquakes
* Severe Storms

into a single environmental intelligence platform with actionable recommendations.

---

# 🌍 Real-world Impact

RiskSense AI helps:

* 🚶 Daily commuters
* 🏃 Runners & cyclists
* 👨‍👩‍👧 Parents planning outdoor activities
* 🎪 Event organizers
* ✈️ Travelers
* 👴 Elderly & respiratory-sensitive individuals

Instead of checking several apps, users receive one clear recommendation within seconds.

---

# 🎥 Demo Video

![Demo Video](Images/Risk-Sense-AI-Demo.mp4)

---

# 📸 Application Screenshots

## Home Page

![Home Page](Images/home-page.png)

---

## Environmental Risk Score

![Risk Score](Images/risk-score.png)

---

## Weather and Interactive Hazard Map

![Weather and Interactive Hazard Map](Images/hazard-map.png)

---

## Air Quality and AI Summary

![Air Quality and AI Summary](Images/air-quality.png)

---

## Forecast and Score Contribution

![Forecast and Score Contribution](Images/forecast-score-contribution.png)

---

## Weather Conditions

![Weather Conditions](Images/weather-conditions.png)

---

# ✨ Key Features

* 🌍 Worldwide location search
* ⚠️ Unified 0–100 Environmental Risk Score
* 🔥 Multi-hazard monitoring
* 🌫 Air Quality Dashboard
* 🌦 Live Weather Monitoring
* 🌍 Interactive Hazard Map
* 📈 6-hour & 12-hour Machine Learning Forecast
* 🤖 Gemini-powered AI explanations
* ⚡ Live pipeline progress (Server-Sent Events)
* ☁️ Google Cloud deployment
* 🚀 NVIDIA RAPIDS GPU acceleration
* 📊 Interactive dashboard with charts and maps

---

# 🔄 How It Works

1. User searches for any location.
2. The application geocodes the location.
3. Environmental data is collected from NASA EONET, OpenWeatherMap, and USGS.
4. Data is cleaned, validated, and transformed.
5. NVIDIA RAPIDS cuDF accelerates feature engineering.
6. The risk engine calculates a 0–100 environmental risk score.
7. A Random Forest model predicts the next 6-hour and 12-hour risk.
8. Gemini generates a natural-language explanation.
9. Results are displayed on an interactive dashboard.

# 📊 Data Sources

| Source                    | Data                                          |
| ------------------------- | --------------------------------------------- |
| NASA EONET                | Wildfires, Volcanoes, Storms, Natural Hazards |
| OpenWeatherMap            | Weather Forecast & Air Quality                |
| USGS                      | Earthquake Data                               |
| Nominatim (OpenStreetMap) | Worldwide Geocoding                           |

---

# 🏗️ Solution Architecture

```mermaid
flowchart TD

A[User Search]
--> B[Nominatim]

B --> C[NASA EONET]
B --> D[OpenWeatherMap]
B --> E[USGS]

C --> F[Data Ingestion]
D --> F
E --> F

F --> G[Cloud Storage]

G --> H[BigQuery]

H --> I[NVIDIA RAPIDS cuDF]

I --> J[Risk Engine]

J --> K[Random Forest]

K --> L[Gemini]

L --> M[FastAPI Dashboard]

M --> N[Cloud Run]
```


---

# ⚙️ Technology Stack

## ☁️ Google Cloud

* Cloud Run
* BigQuery
* Cloud Storage
* Gemini API

## 🚀 NVIDIA

* RAPIDS cuDF
* cudf.pandas

## 🤖 Machine Learning

* Random Forest Regression
* Scikit-learn

## 🌐 Backend

* Python
* FastAPI
* Uvicorn

## 📊 Frontend

* Jinja2
* Tailwind CSS
* Plotly
* Leaflet

---

## 🔄 End-to-End Decision Pipeline

```mermaid
flowchart LR

A["👤 User<br/>Search Location"]

--> B["🌍 Geocoding"]

--> C["🌦️ Weather
🌫️ Air Quality
🔥 Hazards
🌍 Earthquakes"]

--> D["☁️ Cloud Storage"]

--> E["📊 BigQuery"]

--> F["⚡ NVIDIA RAPIDS cuDF
Feature Engineering"]

--> G["🎯 Risk Scoring
0–100"]

--> H["📈 Random Forest
6h & 12h Forecast"]

--> I["🤖 Gemini AI
Recommendation"]

--> J["📊 Interactive Dashboard"]
```
---

# 📂 Project Structure

```text
project-root/
│
├── configs/
├── src/
│    ├── api/     # FastAPI routes and templates
│    ├── ingestion/    # API connectors
│    ├── features/     # Feature engineering
│    ├── model/        # ML forecasting
│    ├── explain/      # Gemini integration
│    ├── dashboard/    # Visualization
│
├── data/
├── sql/
├── Images/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 🌐 API Endpoints

| Endpoint                | Description                    |
| ----------------------- | ------------------------------ |
| GET /                   | Dashboard                      |
| GET /map                | Interactive Hazard Map         |
| GET /forecast           | Risk Forecast                  |
| GET /explanation        | AI Recommendation              |
| GET /benchmark          | Performance Comparison         |
| GET /api/current        | Current Risk Score             |
| GET /api/forecast       | Forecast JSON                  |
| POST /api/refresh       | Refresh Environmental Pipeline |
| GET /api/refresh/stream | Live Pipeline Progress         |

---

# ☁️ Cloud Deployment

RiskSense AI is fully deployed on **Google Cloud Run**.

Google Cloud services used:

* ✅ Cloud Run
* ✅ BigQuery
* ✅ Cloud Storage
* ✅ Gemini API

The application automatically provisions datasets, stores processed environmental data in BigQuery and Cloud Storage, and scales serverlessly based on user demand.

---

# 🚀 NVIDIA Acceleration

RiskSense AI uses **NVIDIA RAPIDS cuDF (cudf.pandas)** to accelerate feature engineering and data processing.

With a single configuration change (`CUDF_USE_PANDAS=true`), the application can execute pandas-based workloads on NVIDIA GPUs, significantly reducing processing time for large environmental datasets while maintaining the same codebase.

---

# 🎯 Why RiskSense AI?

✅ Real-world problem

✅ Decision-support application

✅ Multi-source environmental analytics

✅ Machine Learning forecasting

✅ Explainable AI recommendations

✅ Google Cloud native deployment

✅ NVIDIA GPU acceleration

---

# 🚀 Future Enhancements

* Mobile application
* Personalized alerts
* Historical trend analytics
* Vertex AI integration
* IoT sensor support
* Real-time streaming updates
* User authentication
* Multi-language AI recommendations

---

# 📄 License

This project was developed as part of the **Gen AI Academy APAC Edition Prototype Submission** and is intended for educational, research, and demonstration purposes.
