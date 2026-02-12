"""
Convert JSONL dataset to Parquet format for HuggingFace datasets

This script converts the JSONL file to Parquet format which will load correctly
with features: ['prompt', 'completion', 'answer', 'reward']

Format transformation:
- prompt: [{'content': '...'}] or {'content': '...'} -> [{'role': 'user', 'content': '...'}]
- completions: [{'content': '...'}] or {'content': '...'} -> [{'role': 'assistant', 'content': '...'}] (renamed to 'completion')

Both prompt and completion are converted to lists of message dictionaries with 'role' and 'content' keys.
"""

import json
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd


def convert_jsonl_to_parquet(input_file: str, output_dir: str = None):
    """
    Convert JSONL file to Parquet format that loads correctly with datasets library.
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save Parquet files (default: creates 'parquet_dataset' directory)
    """
    print(f"Loading JSONL file: {input_file}")
    
    # Load directly using datasets library
    print("Loading dataset from JSONL...")
    dataset = load_dataset("json", data_files=input_file, split="train")
    
    print(f"\nOriginal dataset:")
    print(f"Features: {list(dataset.features.keys())}")
    print(f"Number of rows: {len(dataset)}")
    
    # Display sample
    if len(dataset) > 0:
        print("\nSample row:")
        sample = dataset[0]
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Prompt type: {type(sample.get('prompt'))}")
        print(f"  Completions type: {type(sample.get('completions'))}")
        print(f"  Answer: {sample.get('answer', 'N/A')[:50]}...")
        print(f"  Reward: {sample.get('reward', 'N/A')}")
    
    # Transform format: convert prompt and completions to lists of message dictionaries
    print("\nTransforming format to list of message dictionaries...")
    print("  - prompt: [{'content': '...'}] or {'content': '...'} -> [{'role': 'user', 'content': '...'}]")
    print("  - completions: [{'content': '...'}] or {'content': '...'} -> [{'role': 'assistant', 'content': '...'}]")
    
    def transform_format(example):
        """Transform prompt and completions to lists of message dictionaries with roles."""
        # Transform prompt: ensure it's a list with role
        if "prompt" in example:
            prompt_data = example["prompt"]
            if isinstance(prompt_data, list):
                # If it's already a list, ensure each item has role
                if len(prompt_data) > 0:
                    if "role" not in prompt_data[0]:
                        # Add role to first item
                        prompt_data[0] = {"role": "user", "content": prompt_data[0].get("content", "")}
                    example["prompt"] = prompt_data
                else:
                    # Empty list, create default
                    example["prompt"] = [{"role": "user", "content": ""}]
            elif isinstance(prompt_data, dict):
                # Single dict, convert to list
                if "role" not in prompt_data:
                    example["prompt"] = [{"role": "user", "content": prompt_data.get("content", "")}]
                else:
                    example["prompt"] = [prompt_data]
            else:
                # Fallback: create list with content
                example["prompt"] = [{"role": "user", "content": str(prompt_data)}]
        
        # Transform completions: ensure it's a list with role, and rename to completion
        if "completions" in example:
            completions_data = example["completions"]
            if isinstance(completions_data, list):
                # If it's already a list, ensure each item has role
                if len(completions_data) > 0:
                    if "role" not in completions_data[0]:
                        # Add role to first item
                        completions_data[0] = {"role": "assistant", "content": completions_data[0].get("content", "")}
                    example["completion"] = completions_data
                else:
                    # Empty list, create default
                    example["completion"] = [{"role": "assistant", "content": ""}]
            elif isinstance(completions_data, dict):
                # Single dict, convert to list
                if "role" not in completions_data:
                    example["completion"] = [{"role": "assistant", "content": completions_data.get("content", "")}]
                else:
                    example["completion"] = [completions_data]
            else:
                # Fallback: create list with content
                example["completion"] = [{"role": "assistant", "content": str(completions_data)}]
        
        # Remove old completions column if it still exists
        if "completions" in example:
            del example["completions"]
        
        return example
    
    # Apply transformation
    dataset = dataset.map(transform_format, remove_columns=["completions"] if "completions" in dataset.features else [])
    print(f"✅ Format transformed")
    
    print(f"Updated features: {list(dataset.features.keys())}")
    
    # Verify the transformation worked
    if "completion" not in dataset.features:
        print(f"❌ Error: Transformation failed! 'completion' not in features")
        print(f"   Current features: {list(dataset.features.keys())}")
        return None
    
    # Display sample of transformed data
    if len(dataset) > 0:
        print("\nSample transformed row:")
        sample = dataset[0]
        prompt = sample.get('prompt', [])
        completion = sample.get('completion', [])
        
        if isinstance(prompt, list) and len(prompt) > 0:
            print(f"  Prompt: {prompt[0]}")
        else:
            print(f"  Prompt: {prompt}")
        
        if isinstance(completion, list) and len(completion) > 0:
            completion_content = completion[0].get('content', '')[:100]
            print(f"  Completion (first 100 chars): {completion_content}...")
        else:
            print(f"  Completion: {completion}")
        
        print(f"  Answer: {sample.get('answer', 'N/A')[:50]}...")
        print(f"  Reward: {sample.get('reward', 'N/A')}")
    
    # Create DatasetDict with renamed dataset
    dataset_dict = DatasetDict({
        'train': dataset
    })
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(input_file).parent / "parquet_dataset"
    else:
        output_dir = Path(output_dir)
    
    # Remove old directory if it exists to ensure clean conversion
    if output_dir.exists():
        print(f"\n⚠️  Output directory already exists: {output_dir}")
        print(f"   Removing old directory to ensure clean conversion...")
        import shutil
        shutil.rmtree(output_dir)
        print(f"   ✅ Old directory removed")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Saving to Parquet format in: {output_dir}")
    print(f"{'='*80}")
    print(f"Features to be saved: {list(dataset.features.keys())}")
    
    # Method 1: Save using save_to_disk (creates Arrow format)
    print("\nSaving dataset using save_to_disk...")
    dataset_dict.save_to_disk(str(output_dir))
    print(f"✅ Dataset saved to: {output_dir}")
    
    # Method 2: Also save as Parquet files directly for better compatibility
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to pandas and save as Parquet
    print("\nAlso saving as Parquet files for direct loading...")
    df = dataset.to_pandas()
    parquet_file = train_dir / "data.parquet"
    
    # Save as Parquet
    df.to_parquet(parquet_file, engine='pyarrow', index=False)
    print(f"✅ Parquet file saved to: {parquet_file}")
    
    # Verify the saved dataset has the correct column name
    print(f"\nVerifying saved dataset structure...")
    saved_info_file = Path(output_dir) / "train" / "dataset_info.json"
    if saved_info_file.exists():
        import json
        with open(saved_info_file, 'r') as f:
            saved_info = json.load(f)
            saved_features = list(saved_info.get('features', {}).keys())
            print(f"Saved features (from dataset_info.json): {saved_features}")
            if "completion" in saved_features:
                print(f"✅ 'completion' is correctly saved")
            elif "completions" in saved_features:
                print(f"⚠️  Warning: 'completions' (plural) is still in saved file")
    
    # Also verify Parquet file
    print(f"\nVerifying Parquet file columns...")
    import pyarrow.parquet as pq
    parquet_table = pq.read_table(parquet_file)
    parquet_columns = parquet_table.column_names
    print(f"Parquet file columns: {parquet_columns}")
    if "completion" in parquet_columns:
        print(f"✅ 'completion' is in Parquet file")
    elif "completions" in parquet_columns:
        print(f"⚠️  Warning: 'completions' (plural) is in Parquet file")
    
    print(f"✅ Dataset saved to Parquet format in: {output_dir}")
    
    # Verify by reloading
    print(f"\n{'='*80}")
    print("Verifying saved dataset...")
    print(f"{'='*80}")
    
    try:
        reloaded = load_dataset(str(output_dir))
        
        print(f"\nReloaded dataset:")
        print(f"Type: {type(reloaded)}")
        
        # Check if we have the train split
        if 'train' in reloaded:
            train_features = list(reloaded['train'].features.keys())
            print(f"Features: {train_features}")
            print(f"Number of rows: {len(reloaded['train'])}")
            print(f"\nDataset structure:")
            print(reloaded)
            
            # Verify sample data - only if we have the correct features
            if len(reloaded['train']) > 0 and 'prompt' in train_features:
                print(f"\n{'='*80}")
                print("Verification - Sample row from reloaded dataset:")
                print(f"{'='*80}")
                sample = reloaded['train'][0]
                
                # Check prompt format (should be a list)
                prompt = sample.get('prompt', [])
                if isinstance(prompt, list) and len(prompt) > 0:
                    prompt_msg = prompt[0]
                    print(f"Prompt (list format): {prompt_msg}")
                    print(f"  Role: {prompt_msg.get('role', 'N/A')}")
                    print(f"  Content (first 100 chars): {prompt_msg.get('content', '')[:100]}...")
                elif isinstance(prompt, dict):
                    print(f"Prompt (dict format - will be converted): {prompt}")
                else:
                    print(f"Prompt: {prompt}")
                
                # Check completion format (should be a list)
                if 'completion' in sample:
                    completion = sample['completion']
                    if isinstance(completion, list) and len(completion) > 0:
                        completion_msg = completion[0]
                        print(f"Completion (list format): {completion_msg}")
                        print(f"  Role: {completion_msg.get('role', 'N/A')}")
                        print(f"  Content (first 100 chars): {completion_msg.get('content', '')[:100]}...")
                    elif isinstance(completion, dict):
                        print(f"Completion (dict format - will be converted): {completion}")
                    else:
                        print(f"Completion: {completion}")
                elif 'completions' in sample:
                    completions = sample['completions']
                    if isinstance(completions, list) and len(completions) > 0:
                        print(f"Completions (first 100 chars): {completions[0].get('content', '')[:100]}...")
                    else:
                        print(f"Completions: {completions}")
                
                print(f"Answer: {sample.get('answer', 'N/A')}")
                print(f"Reward: {sample.get('reward', 'N/A')}")
            else:
                print(f"\n⚠️  Warning: Reloaded dataset doesn't have expected features.")
                print(f"   Expected: ['prompt', 'completion', 'answer', 'reward']")
                print(f"   Got: {train_features}")
                print(f"\n   This might be a loading issue. Try loading directly from JSONL instead.")
        else:
            print(f"⚠️  Warning: No 'train' split found in reloaded dataset")
            print(f"   Available splits: {list(reloaded.keys())}")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify reloaded dataset: {e}")
        print(f"   The dataset was saved, but verification failed.")
        print(f"   You can try loading it manually to check.")
    
    print(f"\n{'='*80}")
    print("✅ Conversion complete!")
    print(f"{'='*80}")
    print(f"\n{'='*80}")
    print("HOW TO LOAD THE DATASET:")
    print(f"{'='*80}")
    print(f"\nOption 1: Load from Parquet file (RECOMMENDED):")
    print(f"  from datasets import load_dataset")
    parquet_file_path = output_dir / "train" / "data.parquet"
    print(f"  ds = load_dataset('parquet', data_files='{parquet_file_path}', split='train')")
    print(f"  # Or as DatasetDict:")
    print(f"  ds = load_dataset('parquet', data_files='{parquet_file_path}')")
    
    print(f"\nOption 2: Load directly from JSONL (ALWAYS WORKS):")
    print(f"  from datasets import load_dataset")
    print(f"  ds = load_dataset('json', data_files='{input_file}', split='train')")
    print(f"  # Then rename if needed:")
    print(f"  ds = ds.rename_column('completions', 'completion')")
    
    print(f"\nOption 3: Load from saved directory (may have issues):")
    print(f"  from datasets import load_dataset")
    print(f"  ds = load_dataset('{output_dir}')")
    print(f"  # If this shows wrong features, use Option 1 or 2 above")
    
    print(f"\nExpected output:")
    print(f"  DatasetDict({{")
    print(f"      train: Dataset({{")
    print(f"          features: ['prompt', 'completion', 'answer', 'reward'],")
    print(f"          num_rows: {len(dataset)}")
    print(f"      }})")
    print(f"  }})")
    print(f"\nFormat:")
    print(f"  - prompt: [{{'role': 'user', 'content': '...'}}]")
    print(f"  - completion: [{{'role': 'assistant', 'content': '...'}}]")
    
    return dataset_dict


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert JSONL to Parquet format for HuggingFace datasets')
    parser.add_argument('--input', type=str, default='llm_training_dataset.jsonl',
                       help='Input JSONL file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for Parquet files (default: parquet_dataset/)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return 1
    
    try:
        convert_jsonl_to_parquet(str(input_path), args.output)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

