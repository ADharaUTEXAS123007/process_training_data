"""
Convert JSONL dataset to HuggingFace datasets format
This script converts the existing JSONL file to a format compatible with the datasets library.
"""

import json
from pathlib import Path
from datasets import Dataset, DatasetDict
import pandas as pd


def load_jsonl(file_path: str):
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def convert_to_datasets_format(input_file: str, output_dir: str = None):
    """
    Convert JSONL file to HuggingFace datasets format.
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save the dataset (optional)
    """
    print(f"Loading JSONL file: {input_file}")
    
    # Load directly using datasets library (recommended method)
    from datasets import load_dataset
    
    print("Loading dataset using datasets library...")
    dataset_dict = load_dataset("json", data_files=input_file, split="train")
    
    # Wrap in DatasetDict for consistency
    if not isinstance(dataset_dict, DatasetDict):
        dataset_dict = DatasetDict({'train': dataset_dict})
    
    print(f"\nDataset loaded:")
    print(f"Features: {list(dataset_dict['train'].features.keys())}")
    print(f"Number of rows: {len(dataset_dict['train'])}")
    
    # Display sample
    if len(dataset_dict['train']) > 0:
        print("\nSample row:")
        sample = dataset_dict['train'][0]
        print(f"Keys: {list(sample.keys())}")
        print(f"Prompt: {sample.get('prompt', 'N/A')[:100]}...")
        print(f"Answer: {sample.get('answer', 'N/A')}")
        print(f"Reward: {sample.get('reward', 'N/A')}")
    
    # Save if output directory is provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving dataset to: {output_path}")
        dataset_dict.save_to_disk(str(output_path))
        print(f"✅ Dataset saved to {output_path}")
        print("\nTo load the saved dataset, use:")
        print(f"  from datasets import load_dataset")
        print(f"  ds = load_dataset('{output_path}')")
    
    return dataset_dict


def verify_format(data: list):
    """Verify that the data matches the expected format."""
    required_keys = ['prompt', 'completions', 'answer', 'reward']
    errors = []
    
    for idx, row in enumerate(data):
        # Check required keys
        for key in required_keys:
            if key not in row:
                errors.append(f"Row {idx}: Missing key '{key}'")
        
        # Check prompt format
        if 'prompt' in row:
            if not isinstance(row['prompt'], list):
                errors.append(f"Row {idx}: 'prompt' should be a list, got {type(row['prompt'])}")
            elif len(row['prompt']) > 0 and 'content' not in row['prompt'][0]:
                errors.append(f"Row {idx}: 'prompt' list items should have 'content' key")
        
        # Check completions format
        if 'completions' in row:
            if not isinstance(row['completions'], list):
                errors.append(f"Row {idx}: 'completions' should be a list, got {type(row['completions'])}")
            elif len(row['completions']) > 0 and 'content' not in row['completions'][0]:
                errors.append(f"Row {idx}: 'completions' list items should have 'content' key")
        
        # Check answer format
        if 'answer' in row and not isinstance(row['answer'], str):
            errors.append(f"Row {idx}: 'answer' should be a string, got {type(row['answer'])}")
        
        # Check reward format
        if 'reward' in row and not isinstance(row['reward'], (int, float)):
            errors.append(f"Row {idx}: 'reward' should be a number, got {type(row['reward'])}")
    
    if errors:
        print(f"\n⚠️  Found {len(errors)} format errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False
    else:
        print("\n✅ All rows match the expected format!")
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert JSONL to HuggingFace datasets format')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSONL file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for saved dataset (optional)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify format before conversion')
    
    args = parser.parse_args()
    
    # Load and verify
    data = load_jsonl(args.input)
    
    if args.verify:
        print("Verifying format...")
        if not verify_format(data):
            print("\n❌ Format verification failed. Please fix the issues above.")
            return 1
    
    # Convert
    dataset_dict = convert_to_datasets_format(args.input, args.output)
    
    print("\n✅ Conversion complete!")
    print("\nYou can load the dataset directly from JSONL like this:")
    print("  from datasets import load_dataset")
    print(f"  ds = load_dataset('json', data_files='{args.input}', split='train')")
    print("\nOr if you saved it to disk:")
    if args.output:
        print(f"  ds = load_dataset('{args.output}')")
    
    return 0


if __name__ == "__main__":
    exit(main())

