import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("AgriPrice_Chained_XGB_Final")

#feature engineering function to add time features

def add_advanced_features(df):
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df = df.sort_values(['Commodity', 'Market', 'Date']).reset_index(drop=True)
    df['Month']      = df['Date'].dt.month
    df['Year']       = df['Date'].dt.year
    df['DayOfWeek']  = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

    # Log transformation stabilizes the high price variance
    df['Log_Wholesale'] = np.log1p(df['Wholesale'])
    df['Log_Retail']    = np.log1p(df['Retail'])

    return df

def train_models():
    data_path = os.path.join('Dataset', 'kamis_master_final.csv')
    df = pd.read_csv(data_path)

    # Clean and fill gaps
    df['Wholesale'] = df.groupby(['Commodity', 'Market'])['Wholesale'].transform(
        lambda x: x.fillna(x.median())
    )
    df['Wholesale'] = df['Wholesale'].fillna(df['Wholesale'].median())
    df = df.dropna(subset=['Retail', 'County'])

    df = add_advanced_features(df)
    print(f"Rows after feature engineering: {len(df)}")

    categorical_cols = ['Commodity', 'Classification', 'Market', 'County', 'Unit']
    time_features = ['Month', 'Year', 'DayOfWeek', 'WeekOfYear']

    # ================================================================
    # STAGE 1: Wholesale Model (Independent Features only)
    # ================================================================
    ws_features = categorical_cols + time_features

    X_ws = df[ws_features]
    y_ws = df['Log_Wholesale']

    X_train_ws, X_test_ws, y_train_ws, y_test_ws = train_test_split(
        X_ws, y_ws, test_size=0.2, random_state=42
    )

    enc_ws = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_ws_enc = X_train_ws.copy()
    X_test_ws_enc  = X_test_ws.copy()
    X_train_ws_enc[categorical_cols] = enc_ws.fit_transform(X_train_ws[categorical_cols].astype(str))
    X_test_ws_enc[categorical_cols]  = enc_ws.transform(X_test_ws[categorical_cols].astype(str))
    joblib.dump(enc_ws, "encoder_wholesale.joblib")

    with mlflow.start_run(run_name="Stage1_Wholesale_XGB"):
        model_ws = XGBRegressor(
            n_estimators=1000, max_depth=10, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=50, random_state=42, n_jobs=-1
        )
        model_ws.fit(
            X_train_ws_enc, y_train_ws,
            eval_set=[(X_test_ws_enc, y_test_ws)],
            verbose=False
        )

        preds_ws = np.expm1(model_ws.predict(X_test_ws_enc))
        rmse_ws  = np.sqrt(mean_squared_error(np.expm1(y_test_ws), preds_ws))
        r2_ws    = r2_score(np.expm1(y_test_ws), preds_ws)

        mlflow.log_metric("r2", r2_ws)
        mlflow.sklearn.log_model(model_ws, "xgb_wholesale", registered_model_name="AgriPrice_XGB_Wholesale")
        joblib.dump(model_ws, "model_wholesale.joblib")

        print("-" * 40)
        print(f"Wholesale Model Stage 1 — RMSE: {rmse_ws:.2f} | R2: {r2_ws:.4f}")
        print("-" * 40)

    # ================================================================
    # STAGE 2: Retail Model (Uses Wholesale as a high-fidelity signal)
    # ================================================================
    rt_features = categorical_cols + time_features + ['Wholesale']

    X_rt = df[rt_features]
    y_rt = df['Log_Retail']

    X_train_rt, X_test_rt, y_train_rt, y_test_rt = train_test_split(
        X_rt, y_rt, test_size=0.2, random_state=42
    )

    enc_rt = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_rt_enc = X_train_rt.copy()
    X_test_rt_enc  = X_test_rt.copy()
    X_train_rt_enc[categorical_cols] = enc_rt.fit_transform(X_train_rt[categorical_cols].astype(str))
    X_test_rt_enc[categorical_cols]  = enc_rt.transform(X_test_rt[categorical_cols].astype(str))
    joblib.dump(enc_rt, "encoder_retail.joblib")

    with mlflow.start_run(run_name="Stage2_Retail_XGB"):
        model_rt = XGBRegressor(
            n_estimators=1000, max_depth=10, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=50, random_state=42, n_jobs=-1
        )
        model_rt.fit(
            X_train_rt_enc, y_train_rt,
            eval_set=[(X_test_rt_enc, y_test_rt)],
            verbose=False
        )

        preds_rt = np.expm1(model_rt.predict(X_test_rt_enc))
        rmse_rt  = np.sqrt(mean_squared_error(np.expm1(y_test_rt), preds_rt))
        r2_rt    = r2_score(np.expm1(y_test_rt), preds_rt)

        mlflow.log_metric("r2", r2_rt)
        mlflow.sklearn.log_model(model_rt, "xgb_retail", registered_model_name="AgriPrice_XGB_Retail")
        joblib.dump(model_rt, "model_retail.joblib")

        print("-" * 40)
        print(f"Retail Model Stage 2 — RMSE: {rmse_rt:.2f} | R2: {r2_rt:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    train_models()