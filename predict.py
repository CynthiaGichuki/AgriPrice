import mlflow.pyfunc
import pandas as pd
import numpy as np
import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Load Artifacts
encoders = joblib.load("label_encoders.pkl")

# 2. Load the best model from the registry
# Change 'AgriPrice_RF' to 'AgriPrice_ANN' if you want to switch models!
model_name = "AgriPrice_RF" 
model_uri = f"models:/{model_name}/latest"
model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI(title="AgroPulse: Price Prediction API")

class PriceRequest(BaseModel):
    commodity: str
    classification: str
    market: str
    county: str
    month: int
    day_of_week: int

@app.post("/predict")
def get_prediction(data: PriceRequest):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    
    # Translate words to numbers using saved encoders
    try:
        for col in ['commodity', 'classification', 'market', 'county']:
            le = encoders[col.capitalize()]
            input_df[col] = le.transform([input_dict[col]])
    except Exception as e:
        return {"error": f"Invalid input value: {str(e)}"}
    
    prediction = model.predict(input_df)
    
    return {
        "predicted_retail_price": round(float(prediction[0]), 2),
        "currency": "KES"
    }

@app.get("/")
def home():
    return {"message": "AgroPulse API is live. Go to /docs to test!"}