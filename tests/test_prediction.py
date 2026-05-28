import os
from datetime import datetime

import joblib
import numpy as np
import pytest


@pytest.fixture
def models():
    model_ws = joblib.load("model_wholesale.joblib")
    model_rt = joblib.load("model_retail.joblib")
    return model_ws, model_rt


# ── Prediction Logic ──────────────────────────────────────────────────────────

def test_chained_inference_flow(models):
    """Stage 1 (wholesale) feeds into Stage 2 (retail) without errors."""
    model_ws, model_rt = models

    # 9 features: 5 encoded categoricals + 4 time features
    mock_ws = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 5, 2026, 1, 18]])
    prediction_ws_log = model_ws.predict(mock_ws)
    prediction_ws = np.expm1(prediction_ws_log)[0]

    assert prediction_ws > 0, "Wholesale price should be positive"

    # 10 features: same 9 + predicted wholesale price
    mock_rt = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 5, 2026, 1, 18, prediction_ws]])
    prediction_rt_log = model_rt.predict(mock_rt)
    prediction_rt = float(np.expm1(prediction_rt_log)[0])

    assert isinstance(prediction_rt, float), "Retail prediction should be a float"
    assert prediction_rt > 0, "Retail price should be positive"


# ── Input  Validation ──────────────────────────────────────────────────────────
#sends only 3 features to the wholesale model, which expects 9
def test_wholesale_wrong_feature_count_raises_error(models):
    """Wholesale model must reject input with the wrong number of features."""
    model_ws, _ = models
    too_few_features = np.array([[1.0, 2.0, 3.0]])  # only 3, expects 9
    with pytest.raises(Exception):
        model_ws.predict(too_few_features)


def test_retail_missing_wholesale_feature_raises_error(models):
    """Retail model must reject input that is missing the wholesale price column."""
    _, model_rt = models
    missing_wholesale = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 5, 2026, 1, 18]])  # 9 features, expects 10
    with pytest.raises(Exception):
        model_rt.predict(missing_wholesale)


# ── Output Format and Range ───────────────────────────────────────────────────
#API returns values in a JSON response that needs proper floats.
def test_prediction_output_is_float(models):
    """Both model outputs must be plain Python floats."""
    model_ws, model_rt = models

    mock_ws = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 5, 2026, 1, 18]])
    ws_pred = float(np.expm1(model_ws.predict(mock_ws)[0]))

    mock_rt = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 5, 2026, 1, 18, ws_pred]])
    rt_pred = float(np.expm1(model_rt.predict(mock_rt)[0]))

    assert isinstance(ws_pred, float), "Wholesale prediction must be a float"
    assert isinstance(rt_pred, float), "Retail prediction must be a float"

#realistic price range for Kenyan markets, not negative or astronomically high values.
def test_predictions_within_realistic_kes_range(models):
    """Predicted prices must fall within a realistic Kenyan market range (1–100,000 KES)."""
    model_ws, model_rt = models

    mock_ws = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 5, 2026, 1, 18]])
    ws_pred = float(np.expm1(model_ws.predict(mock_ws)[0]))

    mock_rt = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 5, 2026, 1, 18, ws_pred]])
    rt_pred = float(np.expm1(model_rt.predict(mock_rt)[0]))

    assert 1 < ws_pred < 100_000, f"Wholesale price {ws_pred:.2f} KES is outside a realistic range"
    assert 1 < rt_pred < 100_000, f"Retail price {rt_pred:.2f} KES is outside a realistic range"


# ── Error Handling ────────────────────────────────────────────────────────────

def test_valid_date_format_parses_correctly():
    """The expected DD/MM/YYYY format should parse into correct day, month, year."""
    dt = datetime.strptime("01/06/2026", "%d/%m/%Y")
    assert dt.day == 1
    assert dt.month == 6
    assert dt.year == 2026


def test_wrong_date_format_raises_error():
    """ISO format (YYYY-MM-DD) should fail — predict.py only accepts DD/MM/YYYY."""
    with pytest.raises(ValueError):
        datetime.strptime("2026-06-01", "%d/%m/%Y")


def test_garbage_date_raises_error():
    """Completely invalid date strings should raise a ValueError."""
    with pytest.raises(ValueError):
        datetime.strptime("not-a-date", "%d/%m/%Y")


# ── Infrastructure ────────────────────────────────────────────────────────────

def test_data_folder_exists():
    assert os.path.exists("data/"), "Data folder must exist for monitoring"
