# 🌾 AgriPrice — Agricultural Price Forecasting System

> A production-grade MLOps system for forecasting agricultural retail prices across Kenyan markets using a Two-Stage Chained Regressor to reconstruct latent market signals.

---

## 📌 Overview

AgriPrice addresses price transparency for smallholder farmers by forecasting commodity prices across multiple Kenyan markets. Given a commodity, market, and date, the system predicts both the expected wholesale and retail price in KES.

This project covers the full MLOps lifecycle:

- ✅ Data Selection & Preprocessing
- ✅ Model Training with Experiment Tracking
- ✅ Production Deployment
- ✅ Performance Monitoring

**Pipeline stages:**

```
Data Collection → ETL Merge → Model Training (MLflow) → FastAPI Deployment → PostgreSQL → Performance Monitoring (Grafana)
```

---

## 📂 Project Structure

```
AgriPrice/
├── raw_data/                   # 20 raw Excel files (per commodity)
├── Dataset/
│   └── kamis_master_final.csv  # Merged training dataset
├── data/
│   ├── reference.csv           # Baseline data (80% split) for drift detection
│   └── current_batches/        # Production prediction batches (timestamped)
├── scripts/
│   ├── prepare_reference.py    # Creates baseline reference dataset
│   ├── generate_batch.py       # Simulates production batches
│   └── calculate_metrics.py    # Drift detection + metrics logging to PostgreSQL
├── training_pipeline.py        # Two-stage XGBoost training with MLflow
├── predict.py                  # FastAPI inference server
├── merge_script.py             # ETL: consolidates raw Excel files
├── model_wholesale.joblib      # Trained Stage 1 model
├── model_retail.joblib         # Trained Stage 2 model
├── encoder_wholesale.joblib    # OrdinalEncoder for Stage 1
├── encoder_retail.joblib       # OrdinalEncoder for Stage 2
├── Dockerfile                  # Container for the FastAPI server
└── docker-compose.yml          # PostgreSQL + Adminer + Grafana stack
```

---

## 🔬 Stage 1 — Data Selection & Preparation

**Source:** 20 Excel files from KAMIS (Kenya Agricultural Market Information Service), one per commodity.

**Commodities covered:**
- Vegetables: Tomatoes, Cabbages, Carrots, Broccoli
- Fruits: Apples, Oranges, Mangoes, Grapes, Avocado, Pineapples, Watermelon, Tangerine
- Grains: Wheat, Rice, Dry Maize, Wheat Flour, Red Sorghum
- Legumes: Beans Rosecoco
- Other: Eggs, Sweet Potatoes

**ETL pipeline (`merge_script.py`):**
1. Loads all `.xls` files from `raw_data/`
2. Standardizes column names (renames `Variety` → `Classification` where needed)
3. Cleans price strings (removes currency symbols, extracts unit from values like `"80.00/kg"`)
4. Fills missing Unit/Classification with defaults
5. Drops rows with missing Retail price
6. Outputs `Dataset/kamis_master_final.csv`

**Final dataset columns:** `Date, Commodity, Classification, Market, Unit, Wholesale, Retail, County`

---

## 🔬 Stage 2 — Model Training with Experiment Tracking

### Architecture: Two-Stage Chained XGBoost

The model predicts prices in two sequential steps:

| Stage | Input Features | Target |
|-------|---------------|--------|
| Stage 1 | Commodity, Classification, Market, County, Unit, Month, Year, DayOfWeek, WeekOfYear | Wholesale Price |
| Stage 2 | Same features + **predicted wholesale from Stage 1** | Retail Price |

Using the predicted wholesale as a feature in Stage 2 reflects the real-world dependency: retail prices are derived from wholesale costs.

### Key Preprocessing Decisions

- **Grouped Imputation**: Missing wholesale values (~23%) are imputed using median grouped by Commodity + Market to preserve local market context.
- **Log Transformation**: `np.log1p` applied to prices to stabilize the high variance range (50 KES to 12,000 KES).
- **Categorical Encoding**: `OrdinalEncoder` with unknown category handling (`-1`) — encoders serialized separately to prevent training-serving skew.
- **Temporal Features**: Month, Year, DayOfWeek, WeekOfYear extracted from the date column.

### 🧪 MLflow Experiment Tracking

Run the training pipeline:

```bash
python training_pipeline.py
```

MLflow logs for each run:
- Hyperparameters
- R2 and RMSE metrics for both stages
- Model artifacts registered under `Stage1_Wholesale_XGB` and `Stage2_Retail_XGB`

View the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050
```

Open `http://localhost:5050` to compare runs and inspect artifacts.

Serialized artifacts saved locally:
```
model_wholesale.joblib    encoder_wholesale.joblib
model_retail.joblib       encoder_retail.joblib
```

---

## 🚀 Stage 3 — Production Deployment

### FastAPI Inference Server (`predict.py`)

The prediction API runs the full two-stage pipeline on a single request.

**Endpoint:** `POST /predict`

**Request body:**
```json
{
  "commodity": "Tomatoes",
  "classification": "Standard",
  "market": "Nairobi",
  "county": "Nairobi",
  "unit": "kg",
  "date": "28/04/2026"
}
```

**Response:**
```json
{
  "input_summary": { "commodity": "Tomatoes", "market": "Nairobi", "date": "28/04/2026" },
  "stage_1_wholesale_estimate": 250.50,
  "stage_2_retail_forecast": 450.75,
  "currency": "KES",
  "methodology": "Chained XGBoost Inference (V6)"
}
```

### Running with Docker

Build and run the inference container:

```bash
docker build -t agriprice-api .
docker run -p 8000:8000 agriprice-api
```

API will be available at `http://localhost:8000/predict`.

### Full Stack with Docker Compose

```bash
docker-compose up -d
```

This starts three services:

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | Stores drift metrics and production logs |
| Adminer | 8080 | Web UI to inspect PostgreSQL tables |
| Grafana | 3000 | Dashboards for monitoring metrics |

---

## 📡 Stage 4 — Monitoring Pipeline

### Architecture

```
Production Data (20% holdout)
        ↓
generate_batch.py  →  data/current_batches/batch_*.csv
        ↓
calculate_metrics.py
   ├── KS Test (Drift Detection on Wholesale, Month, Year, DayOfWeek)
   ├── RMSE & R2 (Model Performance)
   └── INSERT → PostgreSQL (agri_monitoring DB)
        ↓
Grafana Dashboards
```

### Scripts

**Prepare reference baseline (run once):**
```bash
python scripts/prepare_reference.py
```
Saves the first 80% of training data as `data/reference.csv`.

**Generate a production batch:**
```bash
python scripts/generate_batch.py
```
Samples 500 rows from the 20% holdout, runs chained inference, saves predictions to `data/current_batches/`.

**Calculate metrics and log to PostgreSQL:**
```bash
python scripts/calculate_metrics.py
```
Runs KS tests comparing the current batch against the reference baseline. Logs results to the `model_metrics` table in PostgreSQL.

### Drift Detection

Uses the **K-S test** (threshold: p < 0.05) on four features:
- `Wholesale` — price distribution
- `Month`, `Year`, `DayOfWeek` — temporal distribution

`share_drifted_features` is recorded per batch. A high share signals production data has shifted from the training distribution and the model may need retraining.

---

## Requirements

```bash
pip install fastapi uvicorn xgboost scikit-learn pandas numpy mlflow joblib psycopg2-binary
```

---

## Quick Start

```bash
# 1. Merge raw data
python merge_script.py

# 2. Train models (logs to MLflow)
python training_pipeline.py

# 3. Start monitoring stack
docker-compose up -d

# 4. Start the API
uvicorn predict:app --host 0.0.0.0 --port 8000

# 5. Prepare reference baseline
python scripts/prepare_reference.py

# 6. Simulate a production batch and compute metrics
python scripts/generate_batch.py
python scripts/calculate_metrics.py
```
---

## 📄 License

This project is developed for academic purposes as part of an AI/MLOps course.

---

## 👤 Authors

**Gauri Nair**
[GitHub](https://github.com/GAURI26NAIR)

**Cynthia Gichuki**
[GitHub](https://github.com/CynthiaGichuki)
