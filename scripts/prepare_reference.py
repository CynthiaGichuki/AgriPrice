import pandas as pd
import os

def prepare():
    # Load your master dataset
    df = pd.read_csv('Dataset/kamis_master_final.csv')
    
    # --- ADD FEATURE ENGINEERING HERE ---
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    # ------------------------------------
    
    # Take the first 80% as your 'Reference' (Baseline)
    reference_size = int(len(df) * 0.8)
    reference = df.iloc[:reference_size]
    
    os.makedirs('data', exist_ok=True)
    reference.to_csv('data/reference.csv', index=False)
    print("✅ Baseline reference.csv created with all time features.")

if __name__ == "__main__":
    prepare()