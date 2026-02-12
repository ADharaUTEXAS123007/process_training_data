"""
Convert JSONL dataset to Parquet format for HuggingFace datasets

This script properly converts JSONL to Parquet files that will load correctly.
"""

import json
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import schema


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
        print(f"  Prompt: {sample.get('prompt')}")
        print(f"  Answer: {sample.get('answer', 'N/A')[:50]}...")
        print(f"  Reward: {sample.get('reward', 'N/A')}")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(input_file).parent / "parquet_dataset"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Converting to Parquet format...")
    print(f"{'='*80}")
    
    # Method 1: Use datasets library's built-in Parquet export
    # This should preserve the structure better
    print("\nSaving using datasets library...")
    
    # Convert to DatasetDict
    dataset_dict = DatasetDict({'train': dataset})
    
    # Save to disk - this creates Arrow files, but we want Parquet
    # So we'll use a different approach
    
    # Method 2: Convert to pandas and save as Parquet, then reload
    print("Converting to pandas DataFrame...")
    df = dataset.to_pandas()
    
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    
    # Save as Parquet using pyarrow
    parquet_file = train_dir / "data.parquet"
    print(f"\nSaving to Parquet file: {parquet_file}")
    
    # Convert DataFrame to PyArrow Table
    table = pa.Table.from_pandas(df)
    
    # Write to Parquet
    pq.write_table(table, parquet_file)
    
    print(f"✅ Saved Parquet file: {parquet_file}")
    
    # Now create a dataset from the Parquet file
    print(f"\n{'='*80}")
    print("Creating dataset from Parquet file...")
    print(f"{'='*80}")
    
    # Load from Parquet
    dataset_from_parquet = load_dataset("parquet", data_files=str(parquet_file), split="train")
    
    print(f"\nDataset loaded from Parquet:")
    print(f"Features: {list(dataset_from_parquet.features.keys())}")
    print(f"Number of rows: {len(dataset_from_parquet)}")
    
    # Verify the structure
    if len(dataset_from_parquet) > 0:
        print("\nSample from Parquet dataset:")
        sample = dataset_from_parquet[0]
        print(f"  Keys: {list(sample.keys())}")
        if 'prompt' in sample:
            print(f"  Prompt: {sample['prompt']}")
        if 'answer' in sample:
            print(f"  Answer: {sample['answer']}")
        if 'reward' in sample:
            print(f"  Reward: {sample['reward']}")
    
    # Save the dataset properly using save_to_disk
    print(f"\nSaving dataset structure to: {output_dir}")
    dataset_dict_parquet = DatasetDict({'train': dataset_from_parquet})
    dataset_dict_parquet.save_to_disk(str(output_dir))
    
    print(f"\n{'='*80}")
    print("✅ Conversion complete!")
    print(f"{'='*80}")
    print(f"\nYou can now load the dataset in two ways:")
    print(f"\n1. Load directly from JSONL (RECOMMENDED - always works):")
    print(f"   from datasets import load_dataset")
    print(f"   ds = load_dataset('json', data_files='{input_file}', split='train')")
    print(f"\n2. Load from Parquet file:")
    print(f"   from datasets import load_dataset")
    print(f"   ds = load_dataset('parquet', data_files='{parquet_file}', split='train')")
    print(f"\n3. Load from saved directory (may have issues):")
    print(f"   ds = load_dataset('{output_dir}')")
    
    return dataset_dict_parquet


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


