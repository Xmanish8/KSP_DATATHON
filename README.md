<div align="center">

# SurakshaAI - KSP Crime Intelligence Platform

### AI-Driven Crime Analytics, Prediction and Visualization

**Datathon 2026 | Challenge 02 | Karnataka State Police**

Transforming raw crime records into actionable intelligence using Machine Learning, Geospatial Analysis, and Interactive Dashboards.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-blue)
![Folium](https://img.shields.io/badge/Folium-Maps-green)
![License](https://img.shields.io/badge/License-MIT-success)
![Status](https://img.shields.io/badge/Status-Active-success)
[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-SurakshaAI-blue?style=for-the-badge)](https://project-rainfall-60080092828.development.catalystserverless.in/app/index.html)
</div>

---

## Overview

**SurakshaAI** is a state-of-the-art crime intelligence platform built for the Karnataka State Police (KSP) SCRB (State Crime Records Bureau). It moves beyond static Excel reports to deliver:

- **Interactive Geospatial Maps** - crime hotspots, heatmaps, and district-level choropleth views
- **AI/ML Risk Scoring** - XGBoost-powered district risk classification (LOW / MEDIUM / HIGH)
- **Crime Forecasting** - Prophet time-series models predicting emerging threats
- **Anomaly Detection** - IsolationForest flagging unusual crime spikes
- **Network Analysis** - crime type co-occurrence graphs via Pyvis
- **7-Page Interactive Dashboard** - built with Streamlit, dark-themed, production-ready
- **REST API** - Flask backend with 10 endpoints for integration

---

## 🌐 Live Demo

Experience **SurakshaAI** in action through our deployed prototype.

| Resource | Link |
|----------|------|
| 🌐 Live Demo | https://project-rainfall-60080092828.development.catalystserverless.in/app/index.html |


> **SurakshaAI** is an AI-powered Crime Intelligence & Predictive Analytics Platform designed for the Karnataka State Police. The platform provides interactive crime dashboards, hotspot detection, AI-based risk prediction, crime forecasting, anomaly detection, and criminal network analysis to support proactive policing.

---

## System Architecture

```
data/raw/          ->   pipeline/ (Steps 01-07)    ->   data/processed/
(19 raw CSVs)           (Run in order)                  (Source of truth CSVs)
                                                               |
                                              +----------------+----------------+
                                              |                                 |
                                           api/                           dashboard/
                                     (Flask REST API)             (Streamlit 7-page UI)
                                      10 endpoints                   All pages functional
```

---

## Project Structure

```
KSP_DATATHON/
|
+-- data/
|   +-- raw/                          <- 19 original NCRB state-level CSVs (never modified)
|   |   +-- district/                 <- 57 district-level IPC CSVs (Karnataka)
|   +-- processed/                    <- All cleaned and ML output files (source of truth)
|       +-- master_state_data.csv     <- 10 datasets merged (350 rows x 47 cols)
|       +-- master_features.csv       <- 12 derived features added (350 x 58)
|       +-- district_features.csv     <- District-level features (435 x 116)
|       +-- risk_scores.csv           <- XGBoost predictions + probabilities
|       +-- hotspot_clusters.csv      <- DBSCAN spatial clusters
|       +-- crime_forecast.csv        <- Prophet 3-year forecast per district
|       +-- anomaly_flagged.csv       <- IsolationForest anomaly flags
|       +-- crime_network_data.json   <- Network graph nodes + edges
|       +-- karnataka_hotspot_map.html<- Interactive Folium cluster map
|       +-- karnataka_heatmap.html    <- Interactive Folium heatmap
|       +-- karnataka_choropleth.html <- Interactive choropleth by risk
|       +-- crime_network.html        <- Pyvis interactive network graph
|       +-- xgb_risk_model.pkl        <- Trained XGBoost model
|       +-- xgb_results.png           <- Feature importance + confusion matrix
|       +-- hotspot_clusters.png      <- Static cluster scatter plot
|       +-- forecast_top5_districts.png
|       +-- anomaly_results.png
|       +-- crime_correlation.png     <- Crime type correlation heatmap
|
+-- pipeline/                         <- Run these IN ORDER (01 to 07)
|   +-- 01_preprocess.py              <- Load and clean 14 NCRB CSVs -> master_state_data.csv
|   +-- 02_feature_engineering.py     <- 12 derived features -> master_features.csv
|   +-- 03_risk_classifier.py         <- XGBoost training -> risk_scores.csv + model
|   +-- 04_hotspot_model.py           <- DBSCAN + 3 Folium maps -> hotspot_clusters.csv
|   +-- 05_time_series_forecast.py    <- Prophet forecast -> crime_forecast.csv
|   +-- 06_anomaly_detection.py       <- IsolationForest -> anomaly_flagged.csv
|   +-- 07_network_builder.py         <- Pyvis graph -> crime_network.html + JSON
|   +-- run_all.py                    <- Master runner (runs all 7 steps)
|
+-- api/                              <- Flask REST API
|   +-- app.py
|   +-- engine/
|   |   +-- loader.py
|   |   +-- risk_engine.py
|   +-- routes/
|       +-- risk.py
|       +-- trends.py
|       +-- hotspots.py
|       +-- network.py
|
+-- dashboard/                        <- Streamlit multi-page UI
|   +-- app.py
|   +-- pages/
|       +-- 1_Overview.py
|       +-- 2_Hotspot_Map.py
|       +-- 3_Risk_Predictor.py
|       +-- 4_Crime_Forecast.py
|       +-- 5_Anomaly_Alerts.py
|       +-- 6_Network_Analysis.py
|       +-- 7_Raw_Data.py
|
+-- cli/                              <- Node.js Catalyst CLI
+-- notebooks/                        <- EDA notebooks
+-- requirements.txt
+-- catalyst.json
+-- README.md
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Core |
| API | Flask 3.x | REST endpoints |
| Dashboard | Streamlit 1.35+ | Interactive UI |
| ML Classification | XGBoost 2.x | Risk prediction |
| ML Clustering | Scikit-learn DBSCAN | Hotspot detection |
| ML Forecasting | Facebook Prophet | Time series |
| ML Anomaly | Scikit-learn IsolationForest | Anomaly flagging |
| Network Graph | NetworkX + Pyvis | Crime link analysis |
| Maps | Folium + Branca | Geospatial visualization |
| Charts | Matplotlib, Seaborn, Plotly | Analytics |
| Cloud | Zoho Catalyst | Serverless deployment |
| CLI | Node.js + Commander.js | Catalyst CLI |

---

## Composite Risk Score (6 Dimensions)

Each state is scored on a **0-100 scale** across 6 dimensions:

| # | Dimension | Weight | Key Data Sources |
|---|---|---|---|
| D1 | Police Complaint Volume | 20 pts | Complaints, cases registered |
| D2 | Violent Crime Severity | 20 pts | Murder, rape, kidnapping |
| D3 | Property and Economic Crime | 15 pts | Auto theft, fraud losses |
| D4 | Police Accountability | 15 pts | HR violations, custodial deaths |
| D5 | Judicial Efficiency | 15 pts | Trial duration, conviction rate |
| D6 | System Backlog and Women's Safety | 15 pts | Pending trials, crimes against women |

---

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/<your-username>/KSP_DATATHON.git
cd KSP_DATATHON

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Run all 7 steps in order (takes 3-5 minutes)
python pipeline/run_all.py

# Or run individual steps
python pipeline/01_preprocess.py
python pipeline/02_feature_engineering.py
python pipeline/03_risk_classifier.py
python pipeline/04_hotspot_model.py
python pipeline/05_time_series_forecast.py
python pipeline/06_anomaly_detection.py
python pipeline/07_network_builder.py

# Resume from a specific step
python pipeline/run_all.py --from 3

# Run only one step
python pipeline/run_all.py --only 4
```

### 3. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open: http://localhost:8501

### 4. Start the Flask API

```bash
python api/app.py
```

API runs at: http://127.0.0.1:5000

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/hello` | GET | Health check + metadata |
| `/risk/<state>?year=` | GET | Composite risk score (0-100) + 6 dimensions |
| `/trends/<state>` | GET | Year-by-year trend for all dimensions |
| `/breakdown/<state>` | GET | Crime category breakdown |
| `/states` | GET | All available states |
| `/top-risky?limit=N` | GET | Top N riskiest states |
| `/compare?states=A,B` | GET | Side-by-side state comparison |
| `/hotspots?state=` | GET | DBSCAN cluster data |
| `/network` | GET | Crime network graph JSON |
| `/forecast/<state>` | GET | Prophet forecast data |

---

## Dashboard Pages

| Page | Description |
|---|---|
| Overview | KPI cards, crime trend, top districts, risk distribution |
| Hotspot Map | Folium interactive maps (cluster, heatmap, choropleth) |
| Risk Predictor | XGBoost district risk prediction with probabilities |
| Crime Forecast | Prophet forecast with confidence bands, emerging threats |
| Anomaly Alerts | IsolationForest anomaly cards with CRITICAL/HIGH severity |
| Network Analysis | Pyvis crime type graph + district MO profiles |
| Raw Data | Filterable table with CSV download |

---

## Pipeline Data Flow

```
data/raw/*.csv
      |
      v
01_preprocess.py ───────────────────> master_state_data.csv
      |
      v
02_feature_engineering.py ──────────> master_features.csv
                                       district_features.csv
      |
      +────────────────────────────> 03_risk_classifier.py
      |                                  risk_scores.csv
      |                                  xgb_risk_model.pkl
      |
      +────────────────────────────> 04_hotspot_model.py
      |                                  hotspot_clusters.csv
      |                                  3x Folium HTML maps
      |
      +────────────────────────────> 05_time_series_forecast.py
      |                                  crime_forecast.csv
      |
      +────────────────────────────> 06_anomaly_detection.py
      |                                  anomaly_flagged.csv
      |
      +────────────────────────────> 07_network_builder.py
                                         crime_network.html
                                         crime_network_data.json
```

---

## Datasets Used

| # | File | Rows | Description |
|---|---|---|---|
| 1 | 25_Complaints_against_police.csv | 350 | Police accountability |
| 2 | 20_Victims_of_rape.csv | 350 | Rape victims by age group |
| 3 | 10_Property_stolen_and_recovered.csv | 350 | Property crime values |
| 4 | 32_Murder_victim_age_sex.csv | 342 | Murder victims |
| 5 | 30_Auto_theft.csv | 344 | Vehicle theft |
| 6 | 31_Serious_fraud.csv | 236 | Fraud by amount bracket |
| 7 | 28_Trial_of_violent_crimes_by_courts.csv | 346 | Court trial outcomes |
| 8 | 36_Police_housing.csv | 348 | Police resource capacity |
| 9 | 42_Cases_under_crime_against_women.csv | 350 | Women crime cases |
| 10 | 43_Arrests_under_crime_against_women.csv | 350 | Women crime arrests |
| + | district/01_District_wise_crimes_committed_IPC | 435 | Karnataka district IPC |

---

## Contributors

| Name | Role |
|---|---|
| Xmanish8 | AI/ML, Analytics, Pipeline, Zoho Catalyst, Frontend and Dashboard |
| SohamFE23 | AI/ML, Analytics, Pipeline, Backend and Data Processing |
| rajeshsahu777 | AI/ML, Analytics, Pipeline |

---

## License

Developed for **Datathon 2026 - Challenge 02 (Karnataka State Police)**

Educational and Research Purposes only.

---

<div align="center">

## SurakshaAI

### Predict - Analyze - Prevent

Built with Python, Flask, Streamlit, XGBoost, Folium, Zoho Catalyst

</div>
