import pandas as pd
import glob
import os

# 1. Setup paths
input_path = 'raw_data' 
output_path = 'Dataset'
os.makedirs(output_path, exist_ok=True)

# 2. Get all Excel files
all_files = glob.glob(os.path.join(input_path, "*.xls*"))

if not all_files:
    print(f"Error: No files found in {input_path} folder!")
else:
    li = []

    for filename in all_files:
        df = pd.read_excel(filename)
        
        # --- CLEANING STEP 1: Headers ---
        df.columns = df.columns.str.strip()
        
        # --- CLEANING STEP 2: Handle Missing Columns ---
        if 'Unit' not in df.columns:
            df['Unit'] = 'Standard Unit'
            
        if 'Variety' in df.columns and 'Classification' not in df.columns:
            df = df.rename(columns={'Variety': 'Classification'})
        elif 'Classification' not in df.columns:
            df['Classification'] = 'Standard'
            
        # Define our final "Power Columns"
        important_cols = ['Date', 'Commodity', 'Classification', 'Market', 'Unit', 'Wholesale', 'Retail', 'County']
        
        # Only keep the ones we need
        existing_cols = [c for c in important_cols if c in df.columns]
        df = df[existing_cols].copy()
        
        # --- CLEANING STEP 3: Trim Data ---
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
        
        li.append(df)

    # 3. Merge all files
    master_df = pd.concat(li, axis=0, ignore_index=True)

    # --- FINAL REFINEMENT ---
    master_df['Classification'] = master_df['Classification'].fillna('Standard').replace(['nan', 'None', '', '-'], 'Standard')
    master_df['Unit'] = master_df['Unit'].fillna('Standard Unit').replace(['nan', 'None', ''], 'Standard Unit')

    # Extract unit from Retail price strings like "80.00/kg" or "450.00/Tray(30)"
    def extract_unit(price_str):
        price_str = str(price_str).strip()
        if '/' in price_str:
            return price_str.split('/', 1)[1].strip()
        return 'Standard Unit'

    # Extract Unit BEFORE numeric conversion while string is intact
    if 'Retail' in master_df.columns:
        master_df['Unit'] = master_df['Retail'].astype(str).apply(extract_unit)
        master_df['Unit'] = master_df['Unit'].replace(['nan', 'None', ''], 'Standard Unit')

    # Convert prices to numeric — strips currency symbols, spaces, and other noise
    for col in ['Retail', 'Wholesale']:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(
                master_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                errors='coerce'
            )

    # Debug info before drop
    print(f"Rows before drop: {len(master_df)}")
    if 'Retail' in master_df.columns:
        print(f"Retail NaN count: {master_df['Retail'].isna().sum()}")
        print(f"Sample Retail values:\n{master_df['Retail'].head(10)}")
    else:
        print("WARNING: 'Retail' column not found in any file!")

    print(f"Unique Units found: {master_df['Unit'].nunique()}")
    print(f"Sample Units:\n{master_df['Unit'].value_counts().head(10)}")

    # Drop rows where Retail is missing
    master_df = master_df.dropna(subset=['Retail'])

    # 4. Save
    final_filename = os.path.join(output_path, 'kamis_master_final.csv')
    master_df.to_csv(final_filename, index=False)

    print("-" * 30)
    print(f"SUCCESS: {final_filename} created!")
    print(f"Total Rows: {len(master_df)}")
    print(f"Unique Units found: {master_df['Unit'].nunique()}")
    print("-" * 30)