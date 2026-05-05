import pandas as pd
import os
import time
import joblib
import numpy as np

def generate():
    # 1. Load the data 
    df = pd.read_csv('Dataset/kamis_master_final.csv')
    
    # FEATURE ENGINEERING 
    # Convert 'Date' string to datetime objects and extract the required features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    
    # Load the serialized artifacts (Models and Encoders)
    model_ws = joblib.load("model_wholesale.joblib")
    enc_ws = joblib.load("encoder_wholesale.joblib")
    model_rt = joblib.load("model_retail.joblib")
    enc_rt = joblib.load("encoder_retail.joblib")

    # Shuffle the entire dataset randomly so commodities are mixed
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 2. Simulate "Production" data (the last 20% of the dataset)
    production_data = df.iloc[int(len(df) * 0.8):].copy()
    
    # Select a micro-batch (n=500) to simulate a real-world evaluation window
    batch = production_data.sample(n=500)

    # 3. RUN THE CHAINED INFERENCE (Stage 1 -> Stage 2)
    # This simulates the production API logic
    categorical_cols = ['Commodity', 'Classification', 'Market', 'County', 'Unit']
    time_features = ['Month', 'Year', 'DayOfWeek', 'WeekOfYear']
    
    # Pre-process: Encode categorical features for the Wholesale model
    X_ws = batch.copy()
    X_ws[categorical_cols] = enc_ws.transform(batch[categorical_cols].astype(str))
    
    # Stage 1: Predict Wholesale (Model was trained on log-transformed data)
    log_ws_pred = model_ws.predict(X_ws[categorical_cols + time_features])
    batch['predicted_wholesale'] = np.expm1(log_ws_pred)

    # Stage 2: Predict Retail
    X_rt = batch.copy()
    X_rt[categorical_cols] = enc_rt.transform(batch[categorical_cols].astype(str))
    
    # Create the input for Stage 2
    rt_input = pd.concat([X_rt[categorical_cols + time_features], batch[['predicted_wholesale']]], axis=1)
    
    # RENAME 'predicted_wholesale' to 'Wholesale' so it matches what the model expects
    rt_input = rt_input.rename(columns={'predicted_wholesale': 'Wholesale'})
    
    # prediction
    log_rt_pred = model_rt.predict(rt_input)
    batch['prediction'] = np.expm1(log_rt_pred) 
    # ------------------------

    # 4. Save the batch to the monitoring folder
    os.makedirs('data/current_batches', exist_ok=True)
    timestamp = int(time.time())
    batch.to_csv(f'data/current_batches/batch_{timestamp}.csv', index=False)
    
    print(f" Production batch_{timestamp}.csv generated successfully.")
    print(f"   Includes 'prediction' column for monitoring analysis.")

if __name__ == "__main__":
    generate()