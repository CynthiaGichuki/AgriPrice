import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.keras
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. Setup MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("AgriPrice_Production_Pipeline")

def train_models():
    # --- Data Loading ---
    data_path = os.path.join('Dataset', 'merged_commodities.csv')
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['RetailUnitPrice', 'Commodity', 'Market'])
    
    # --- Feature Engineering ---
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # --- Encoding ---
    categorical_cols = ['Commodity', 'Classification', 'Market', 'County']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    joblib.dump(encoders, "label_encoders.pkl")
    
    # --- Splitting Data ---
    features = categorical_cols + ['Month', 'DayOfWeek']
    X = df[features]
    y = df['RetailUnitPrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling for the Neural Network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.joblib")

    # --- MODEL 1: Random Forest ---
    with mlflow.start_run(run_name="Random_Forest_Run"):
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        
        mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, preds)))
        mlflow.log_artifact("label_encoders.pkl")
        mlflow.sklearn.log_model(rf, "rf_model", registered_model_name="AgriPrice_RF")
        print("Random Forest Trained.")

    # --- MODEL 2: Deep Learning (ANN) ---
    with mlflow.start_run(run_name="Neural_Network_Run"):
        ann = Sequential([
            Dense(64, activation='relu', input_shape=(len(features),)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1) # Output layer for price
        ])
        ann.compile(optimizer='adam', loss='mse')
        ann.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)
        
        preds_ann = ann.predict(X_test_scaled)
        mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, preds_ann)))
        mlflow.log_artifact("scaler.joblib")
        mlflow.keras.log_model(ann, "ann_model", registered_model_name="AgriPrice_ANN")
        print("Neural Network Trained.")

if __name__ == "__main__":
    train_models()