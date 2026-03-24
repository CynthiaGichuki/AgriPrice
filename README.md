# 🌾 AgriPrice: Predictive Pricing and Market Analytics for Kenyan Smallholder Farmers And Consumers
<img width="979" height="260" alt="image" src="https://github.com/user-attachments/assets/9e3d594f-929d-4619-8088-6d1066b5b4a8" />


AgriPrice is a production-grade MLOps project designed to forecast agricultural retail prices across Kenyan markets. By implementing a **Two-Stage Chained Regressor**, the system reconstructs latent market signals to provide accurate forecasts for both farmers and consumers.

## 🏗️ Architecture: The Chained Inference
The core innovation of this project is the **Stage-to-Stage Handshake**. We simulate the agricultural supply chain by breaking the prediction into two distinct "brains":

1.  **Stage 1 (Wholesale Brain):** An XGBoost regressor that predicts the market wholesale price based on **Commodity, Market, and Seasonality**.
2.  **Stage 2 (Retail Brain):** A second XGBoost regressor that takes the **predicted output from Stage 1** and uses it as a high-fidelity feature to calculate the final consumer price.

## 🛠️ MLOps Highlights
* **Grouped Imputation:** Missing values (approx. 23%) are handled via medians calculated per `Commodity + Market` group to preserve local price context.
* **Log-Target Scaling:** We use `np.log1p` to stabilize high price variance across crops (handling 50 KES items and 12,000 KES items in the same model).
* **Experiment Tracking:** Integrated with **MLflow** for hyperparameter logging, metric auditing, and model versioning.
* **Artifact Management:** Dedicated encoders and models are versioned and saved as `.joblib` files to prevent "Training-Serving Skew."
* **Production API:** Built with **FastAPI**, featuring a logic gate that automatically bridges Stage 1 and Stage 2 during a single request.

## 📁 Project Structure
* `training_pipeline.py` — The engine that trains the chained models and logs to MLflow.
* `predict.py` — FastAPI implementation for real-time, two-stage inference.
* `Dataset/` — Contains the `kamis_master_final.csv`.
* `mlflow.db` & `mlruns/` — The "source of truth" for experiment history.

## 🚦 Getting Started

### 1. Training the Models
```bash
python training_pipeline.py
