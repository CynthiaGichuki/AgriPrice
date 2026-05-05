import os
import glob
from datetime import datetime
import pandas as pd 
import psycopg
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ks_2samp

# 1. Update features to match the AgriPrice project
# We monitor numerical features for distribution drift
FEATURES = ["Wholesale", "Month", "Year", "DayOfWeek"]

def main():
    # load reference data (baseline training set)
    reference = pd.read_csv("data/reference.csv")

    # load latest batch of production logs
    batch_files = glob.glob("data/current_batches/*.csv")
    if not batch_files:
        print("No batch files found.")
        return
    latest_file = max(batch_files, key=os.path.getmtime)
    current = pd.read_csv(latest_file)
    batch_id = os.path.basename(latest_file).replace(".csv", "")

    # 2. Compute Data Drift (Numerical features via K-S test)
    drifted_features = 0
    for feature in FEATURES:
        # KS test detects if the production data distribution has shifted 
        _, p_value = ks_2samp(reference[feature], current[feature])
        if p_value < 0.05: 
            drifted_features += 1

    share_drifted = drifted_features / len(FEATURES)

    # 3. Compute Regression Performance (KES Prices)
    # Use RMSE and R2 
    # 'Retail' is the target, 'prediction' is the Stage 2 output
    rmse = np.sqrt(mean_squared_error(current["Retail"], current["prediction"]))
    r2 = r2_score(current["Retail"], current["prediction"])

    # connect to Postgres using the .env credentials
    conn = psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "agri_monitoring"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "agri_password_2026"),
    )

    # 4. Create Table with Agri-specific columns
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp TIMESTAMP,
                batch_id TEXT,
                batch_size INT,
                num_drifted_features INT,
                share_drifted_features FLOAT,
                rmse FLOAT,
                r2_score FLOAT
            );
        """)

        # insert one row of monitoring data
        cur.execute("""
            INSERT INTO metrics (
                timestamp, batch_id, batch_size, 
                num_drifted_features, share_drifted_features, 
                rmse, r2_score
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            datetime.utcnow(),
            batch_id,
            len(current),
            drifted_features,
            share_drifted,
            rmse,
            r2
        ))

    conn.commit()
    conn.close()
    print(f"AgriPrice metrics saved: RMSE={rmse:.2f}, R2={r2:.4f}, Drift={share_drifted:.2%}")

if __name__ == "__main__":
    main()