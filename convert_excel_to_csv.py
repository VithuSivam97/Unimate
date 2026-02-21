import pandas as pd
import os

excel_path = "backend/data/z_scores 2024.xlsx"
csv_path = "backend/data_fast/z_scores 2024.csv"

if not os.path.exists(excel_path):
    print(f"File not found: {excel_path}")
    exit(1)

try:
    # Read Excel file
    # header=0 implies the first row is the header
    df = pd.read_excel(excel_path)
    
    print(f"Read Excel with shape: {df.shape}")
    
    # Clean up headers
    # The output from inspect showed "MEDICINE", "MEDICINE.1" etc.
    # This suggests the Excel might have sub-headers or multiple columns for the same course (e.g. Merit, District, etc?)
    # OR it's a wide format where courses are repeated. 
    # Let's inspect the first few rows to understand the structure before blind conversion.
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    
    # Save to CSV
    # We'll just dump it for now, the backend might need adjustments if the structure is vastly different
    # But for now, let's get the data in matching the filename expected by the backend
    
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")
        
except Exception as e:
    print(f"Error converting: {e}")
