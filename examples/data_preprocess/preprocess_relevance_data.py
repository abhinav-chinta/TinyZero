# examples/data_preprocess/preprocess_relevance_data.py

import re
import os
import datasets
import argparse
from tqdm import tqdm

# Assuming verl.utils.fs handles both local and potential remote fs like HDFS
from verl.utils.fs import copy, makedirs

# --- Regex patterns to extract parts ---
# Using re.DOTALL to make '.' match newlines as well
QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.DOTALL)
DOCUMENT_RE = re.compile(r"<document>(.*?)</document>", re.DOTALL)
# Find *all* reasoning blocks
REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
# Find *all* relevance values (capturing group)
RELEVANCE_RE = re.compile(r"<relevance>([01])</relevance>", re.DOTALL)
# We need the text *before* the specific question/document starts, which is the instruction block.
# Let's capture everything up to the first <question> tag.
INSTRUCTION_RE = re.compile(r"^(.*?)<question>", re.DOTALL)

def extract_parts(text):
    """Extracts structured information, focusing on the second reasoning/relevance block."""
    question_match = QUESTION_RE.search(text)
    document_match = DOCUMENT_RE.search(text)
    instruction_match = INSTRUCTION_RE.search(text)

    # Find all occurrences
    reasoning_matches = REASONING_RE.findall(text)
    # findall on a group returns only the group content
    relevance_matches = RELEVANCE_RE.findall(text) # Returns list like ['1', '1'] or ['0', '1'] etc.

    # --- Validation ---
    # Need the main structure: instructions, question, document
    if not instruction_match or not question_match or not document_match:
        print(f"Warning: Missing instruction, question, or document in text:\n{text[:500]}...")
        return None

    # Expecting at least two reasoning blocks and two relevance tags: one in the instructions, one for the actual data.
    # We will target the *last* one found as the actual data for this row.
    if len(reasoning_matches) < 1 or len(relevance_matches) < 1:
         # If even one is not found, something is wrong with the format
         print(f"Warning: Could not find required reasoning/relevance blocks in text:\n{text[:500]}...")
         return None
    # Optional: Stricter check if you know there are *always* exactly 2 pairs
    # if len(reasoning_matches) != 2 or len(relevance_matches) != 2:
    #     print(f"Warning: Expected 2 reasoning/relevance blocks, found {len(reasoning_matches)}/{len(relevance_matches)} in text:\n{text[:500]}...")
    #     # Decide whether to skip or try to recover, for now we proceed taking the last one

    # --- Extraction ---
    # Instructions are everything before the <question> tag
    instructions = instruction_match.group(1).strip()

    # The actual question and document for this example
    question = question_match.group(1).strip()
    document = document_match.group(1).strip()

    # Take the *last* found reasoning and relevance as the actual ones for this example
    actual_reasoning = reasoning_matches[-1].strip()
    actual_relevance = int(relevance_matches[-1]) # Convert last found '0' or '1' to int

    return {
        "instructions": instructions, # This includes the example reasoning/relevance
        "question": question,
        "document": document,
        "reasoning": actual_reasoning, # This is the specific reasoning for this row
        "relevance": actual_relevance # This is the specific label for this row
    }

# --- make_map_fn remains largely the same, but ensure it uses the output correctly ---

def make_map_fn(split, data_source):
    """Creates the mapping function for the dataset."""
    def process_fn(example, idx):
        text = example.get('text', '')
        if not text:
            print(f"Warning: Empty text field found at index {idx} in split {split}")
            return None # Or handle appropriately

        extracted = extract_parts(text)
        if extracted is None:
            # Skip rows that couldn't be parsed
             return None

        # --- Construct the prompt for the RL model ---
        # The prompt includes the full instructions (with the example reasoning/relevance),
        # the specific question, and the specific document.
        prompt_content = (
            f"{extracted['instructions']}\n\n"
            f"<question>\n{extracted['question']}\n</question>\n\n"
            f"<document>\n{extracted['document']}\n</document>\n\n"
            # Add the trigger asking the model to generate its reasoning and relevance
            "Please provide your reasoning and relevance decision."
            # Alternative trigger if needed: "<reasoning>" # Start tag might help guide the model
        )

        # --- Structure the output dictionary for verl ---
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user", # Adjust role structure if needed
                "content": prompt_content
            }],
            "ability": "relevance_reasoning", # Descriptive ability tag
            "reward_model": {
                "style": "rule", # Placeholder
                # Use the *actual* relevance extracted (the second one)
                "ground_truth": extracted['relevance']
            },
            "extra_info": {
                'split': split,
                'index': idx,
                # Store the *actual* reasoning extracted (the second one)
                'original_reasoning': extracted['reasoning'],
                'original_question': extracted['question'],
                'original_document': extracted['document'],
            }
        }
        return data
    return process_fn

# --- Main execution block remains the same ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess relevance dataset for veRL GRPO training.")
    parser.add_argument('--local_dir', type=str, default='~/data/relevance_rerankers_verl',
                        help='Local directory to save processed Parquet files.')
    parser.add_argument('--hdfs_dir', type=str, default=None,
                        help='Optional HDFS (or other remote FS) directory to copy data to.')
    parser.add_argument('--dataset_name', type=str,
                        default='sumukshashidhar-testing/reasoning-rerankers-relevance-sft-data',
                        help='Name of the Hugging Face dataset to process.')
    # Add arguments for num_proc, batch_size if needed for large datasets

    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir
    data_source = args.dataset_name # Use the dataset name as the source identifier

    print(f"Loading dataset: {data_source}...")
    # Consider adding cache_dir argument if needed
    raw_dataset = datasets.load_dataset(data_source)

    print("Processing splits...")
    processed_datasets = {}
    for split in raw_dataset.keys(): # Process all available splits (train, test)
        print(f"  Processing split: {split}...")
        # Using map with index. Consider adding num_proc for speed on large datasets.
        processed_ds = raw_dataset[split].map(
            make_map_fn(split, data_source),
            with_indices=True,
            batched=False, # Process row by row
            remove_columns=raw_dataset[split].column_names # Remove original columns
        )
        # Filter out None entries resulting from parsing errors
        processed_ds = processed_ds.filter(lambda x: x is not None)
        processed_datasets[split] = processed_ds
        print(f"  Finished processing {split}. Number of examples: {len(processed_ds)}")


    print(f"Saving processed data locally to: {local_dir}")
    os.makedirs(local_dir, exist_ok=True)

    for split, ds in processed_datasets.items():
        output_path = os.path.join(local_dir, f'{split}.parquet')
        print(f"  Saving {split} split to {output_path}...")
        ds.to_parquet(output_path)
        print(f"  Successfully saved {split}.parquet")

    # Optional: Copy to HDFS or other remote storage
    if hdfs_dir:
        print(f"Copying data from {local_dir} to remote {hdfs_dir}...")
        try:
            makedirs(hdfs_dir) # Ensure remote directory exists
            copy(src=local_dir, dst=hdfs_dir) # Copies the entire directory
            print("Successfully copied data to remote destination.")
        except Exception as e:
            print(f"Error copying data to remote destination: {e}")

    print("Preprocessing finished.")