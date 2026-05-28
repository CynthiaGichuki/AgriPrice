VALID_REQUEST = {
    "commodity": "Maize",
    "classification": "Grain",
    "market": "Nairobi",
    "county": "Nairobi",
    "unit": "KG",
    "date": "01/06/2026"
}


# ── Home Endpoint ─────────────────────────────────────────────────────────────

def test_home_endpoint_is_alive(api_client):
    """GET / should return 200 and confirm the API is active."""
    response = api_client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


# ── Predict Endpoint — Valid Request ─────────────────────────────────────────

def test_predict_valid_request_returns_200(api_client):
    """A well-formed POST /predict request should return HTTP 200."""
    response = api_client.post("/predict", json=VALID_REQUEST)
    assert response.status_code == 200

#checks that the response JSON contains all the expected fields, this is the contract
def test_predict_response_has_required_keys(api_client):
    """Response must contain all expected keys."""
    response = api_client.post("/predict", json=VALID_REQUEST)
    data = response.json()
    assert "stage_1_wholesale_estimate" in data
    assert "stage_2_retail_forecast" in data
    assert "currency" in data
    assert "input_summary" in data

#currency should always be KES, not just any string
def test_predict_currency_is_kes(api_client):
    """Currency in the response should always be KES."""
    response = api_client.post("/predict", json=VALID_REQUEST)
    assert response.json()["currency"] == "KES"

#predicted prices should be positive floats, not negative or non-numeric values
def test_predict_prices_are_positive(api_client):
    """Both predicted prices must be greater than zero."""
    response = api_client.post("/predict", json=VALID_REQUEST)
    data = response.json()
    assert data["stage_1_wholesale_estimate"] > 0
    assert data["stage_2_retail_forecast"] > 0

#the input summary should accurately reflect the request, not just echo back random values
def test_predict_input_summary_echoes_request(api_client):
    """Response should echo back the commodity, market and date that were sent."""
    response = api_client.post("/predict", json=VALID_REQUEST)
    summary = response.json()["input_summary"]
    assert summary["commodity"] == VALID_REQUEST["commodity"]
    assert summary["market"] == VALID_REQUEST["market"]
    assert summary["date"] == VALID_REQUEST["date"]


# ── Predict Endpoint — Error Handling ────────────────────────────────────────
#invalid date formats should be gracefully handled with an error message, not a server crash
def test_predict_invalid_date_returns_error_message(api_client):
    """Wrong date format should return an error message, not crash the API."""
    bad_request = {**VALID_REQUEST, "date": "2026-06-01"}  # ISO format, not DD/MM/YYYY
    response = api_client.post("/predict", json=bad_request)
    assert response.status_code == 200
    assert "error" in response.json()

#server understood request but the data is incomplete or invalid
def test_predict_missing_field_returns_422(api_client):
    """A request missing a required field should return HTTP 422 (Unprocessable Entity)."""
    incomplete_request = {
        "commodity": "Maize",
        "market": "Nairobi"
        # missing classification, county, unit, date
    }
    response = api_client.post("/predict", json=incomplete_request)
    assert response.status_code == 422
