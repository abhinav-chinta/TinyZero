import pandas as pd
import os
import json

# Directory containing parquet files
data_dir = "../../data/relevance_rerankers_verl"

# Find all parquet files in the directory
parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

# Create a dictionary to store results
results = {}

# Load each parquet file and store data
for file in parquet_files:
    file_path = os.path.join(data_dir, file)
    
    try:
        # Load the parquet file
        df = pd.read_parquet(file_path)
        
        # Store info about the dataframe
        file_data = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'first_5_rows': df.head(5).to_dict(orient='records')
        }
        
        results[file] = file_data
        
    except Exception as e:
        results[file] = {'error': str(e)}

# Save results to JSON file
output_file = "parquet_visualization.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"Results saved to {output_file}")