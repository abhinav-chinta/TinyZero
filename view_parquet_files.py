import pandas as pd
import os

# Directory containing parquet files
data_dir = "data/relevance_rerankers_verl"

# Find all parquet files in the directory
parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

# Load each parquet file and print first few rows
for file in parquet_files:
    file_path = os.path.join(data_dir, file)
    print(f"\nReading file: {file_path}")
    
    try:
        # Load the parquet file
        df = pd.read_parquet(file_path)
        
        # Print info about the dataframe
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Print first 5 rows
        print("\nFirst 5 rows:")
        print(df.head(5))
        
    except Exception as e:
        print(f"Error reading {file}: {e}") 