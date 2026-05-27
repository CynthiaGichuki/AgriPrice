import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor


@pytest.fixture(scope="session", autouse=True)
def create_test_artifacts():
    """Train minimal mock models matching the training pipeline feature schema."""
    n = 200
    rng = np.random.default_rng(42)

    cat_df = pd.DataFrame({
        "Commodity":      rng.choice(["Maize", "Wheat", "Rice", "Tomatoes"], n).astype(str),
        "Classification": rng.choice(["Grain", "Cereal", "Vegetable"], n).astype(str),
        "Market":         rng.choice(["Nairobi", "Mombasa", "Kisumu"], n).astype(str),
        "County":         rng.choice(["Nairobi", "Mombasa", "Kisumu", "Nakuru"], n).astype(str),
        "Unit":           rng.choice(["KG", "90KG BAG", "BUNCH"], n).astype(str),
    })
    time_arr = np.column_stack([
        rng.integers(1, 13, n),
        rng.integers(2020, 2026, n),
        rng.integers(0, 7, n),
        rng.integers(1, 53, n),
    ])

    # Wholesale model: 5 encoded categoricals + 4 time features = 9 features
    enc_ws = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_ws = np.hstack([enc_ws.fit_transform(cat_df), time_arr])
    y_ws = np.log1p(rng.uniform(50, 500, n))
    model_ws = XGBRegressor(n_estimators=10, max_depth=2, random_state=42)
    model_ws.fit(X_ws, y_ws)

    # Retail model: same 9 features + wholesale price = 10 features
    enc_rt = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_rt = np.hstack([enc_rt.fit_transform(cat_df), time_arr, np.expm1(y_ws).reshape(-1, 1)])
    y_rt = y_ws + np.log1p(rng.uniform(10, 50, n))
    model_rt = XGBRegressor(n_estimators=10, max_depth=2, random_state=42)
    model_rt.fit(X_rt, y_rt)

    joblib.dump(model_ws, "model_wholesale.joblib")
    joblib.dump(model_rt, "model_retail.joblib")
    joblib.dump(enc_ws, "encoder_wholesale.joblib")
    joblib.dump(enc_rt, "encoder_retail.joblib")
    os.makedirs("data", exist_ok=True)

    yield

    for f in ["model_wholesale.joblib", "model_retail.joblib",
              "encoder_wholesale.joblib", "encoder_retail.joblib"]:
        if os.path.exists(f):
            os.remove(f)
