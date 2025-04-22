import pandas as pd
import os
import json

# Directory containing parquet files
data_dir = "../../data/relevance_rerankers_verl"

# Number of examples to save
num_examples = 5

# Find all parquet files in the directory
parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

# Dictionary to store examples from each file
all_examples = {}

# Load each parquet file and extract examples
for file in parquet_files:
    file_path = os.path.join(data_dir, file)
    filename = os.path.splitext(file)[0]  # Get filename without extension
    
    try:
        # Load the parquet file
        df = pd.read_parquet(file_path)
        
        # Take the first few examples
        examples = df.head(num_examples).to_dict(orient='records')
        all_examples[filename] = examples
        
        print(f"Extracted {len(examples)} examples from {file}")
        
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Save to a JSON file with proper indentation for readability
output_file = "examples_data.json"
with open(output_file, 'w') as f:
    json.dump(all_examples, f, indent=2, default=str)

print(f"\nSaved {sum(len(examples) for examples in all_examples.values())} examples to {output_file}") 