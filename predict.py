import pandas as pd
import numpy as np
import joblib
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

# 1. Setup MLflow Tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(title="AgriPrice: Chained Inference API")

# 2. Load the Artifacts
# These must match the names saved in the training_pipeline.py
model_ws = joblib.load("model_wholesale.joblib")
enc_ws = joblib.load("encoder_wholesale.joblib")
model_rt = joblib.load("model_retail.joblib")
enc_rt = joblib.load("encoder_retail.joblib")

class PriceRequest(BaseModel):
    commodity: str
    classification: str
    market: str
    county: str
    unit: str
    date: str  # Format: "DD/MM/YYYY"

@app.post("/predict")
def get_chained_prediction(data: PriceRequest):
    # --- A. Feature Engineering (DateTime) ---
    try:
        date_dt = datetime.strptime(data.date, "%d/%m/%Y")
        time_features = {
            'Month': date_dt.month,
            'Year': date_dt.year,
            'DayOfWeek': date_dt.weekday(),
            'WeekOfYear': int(date_dt.isocalendar()[1])
        }
    except Exception:
        return {"error": "Invalid date format. Use DD/MM/YYYY"}

    # Base Categorical Data
    cat_list = [data.commodity, data.classification, data.market, data.county, data.unit]
    #categorical_cols = ['Commodity', 'Classification', 'Market', 'County', 'Unit']

    # --- B. STAGE 1: Predict Wholesale ---
    # 1. Encode categories using the Wholesale Encoder
    cat_enc_ws = enc_ws.transform([cat_list])[0]
    
    # 2. Combine encoded cats with time features
    # Order must match training: cat_cols + Month, Year, DayOfWeek, WeekOfYear
    X_ws = np.array([list(cat_enc_ws) + list(time_features.values())])
    
    # 3. Predict Wholesale (Log scale -> Raw)
    log_ws_pred = model_ws.predict(X_ws)[0]
    predicted_wholesale = np.expm1(log_ws_pred)

    # --- C. STAGE 2: Predict Retail ---
    # 1. Encode categories using the Retail Encoder
    cat_enc_rt = enc_rt.transform([cat_list])[0]
    
    # 2. Combine: encoded cats + time features + PREDICTED wholesale
    X_rt = np.array([list(cat_enc_rt) + list(time_features.values()) + [predicted_wholesale]])
    
    # 3. Predict Retail (Log scale -> Raw)
    log_rt_pred = model_rt.predict(X_rt)[0]
    predicted_retail = np.expm1(log_rt_pred)

    # --- D. Return Response ---
    return {
        "input_summary": {
            "commodity": data.commodity,
            "market": data.market,
            "date": data.date
        },
        "stage_1_wholesale_estimate": round(float(predicted_wholesale), 2),
        "stage_2_retail_forecast": round(float(predicted_retail), 2),
        "currency": "KES",
        "methodology": "Chained XGBoost Inference (V6)"
    }

@app.get("/")
def home():
    return {"message": "AgriPrice Chained API is active. Visit /docs for testing."}