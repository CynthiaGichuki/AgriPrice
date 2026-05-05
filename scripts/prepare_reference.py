import pandas as pd
import numpy as np
import os

def prepare():
    # 1. Load master dataset
    df = pd.read_csv('Dataset/kamis_master_final.csv')
    
    # 2. Match Feature Engineering from training_pipeline.py
    # Using 'mixed' format to prevent the parser warnings 
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    
    # 3. Add the Target Transformations
    # This ensures the reference data matches the trained model's internal logic
    df['Log_Wholesale'] = np.log1p(df['Wholesale'])
    df['Log_Retail'] = np.log1p(df['Retail'])

    # Shuffle the entire dataset randomly so commodities are mixed
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 4. Take the first 80% as the 'Reference' (Baseline)
    # This represents the "stateless" training data
    reference_size = int(len(df) * 0.8)
    reference = df.iloc[:reference_size]
    
    # 5. Save Artifact
    os.makedirs('data', exist_ok=True)
    reference.to_csv('data/reference.csv', index=False)
    print("Baseline reference.csv created with all time and log features.")

if __name__ == "__main__":
    prepare()