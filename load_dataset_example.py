"""
Example: How to load the LLM training dataset correctly

The dataset should be loaded directly from JSONL file, not from a saved directory.
"""

from datasets import load_dataset

# Method 1: Load directly from JSONL file (RECOMMENDED)
# This preserves all features: prompt, completions, answer, reward
print("="*80)
print("Method 1: Load directly from JSONL file")
print("="*80)

ds = load_dataset("json", data_files="./llm_training_dataset.jsonl", split="train")

print(f"\nDataset loaded:")
print(f"Features: {list(ds.features.keys())}")
print(f"Number of rows: {len(ds)}")
print(f"\nDataset structure:")
print(ds)

# Access a sample row
if len(ds) > 0:
    print("\n" + "="*80)
    print("Sample row:")
    print("="*80)
    sample = ds[0]
    print(f"Prompt: {sample['prompt']}")
    print(f"\nCompletions (first 200 chars): {sample['completions'][0]['content'][:200]}...")
    print(f"\nAnswer: {sample['answer']}")
    print(f"\nReward: {sample['reward']}")

# Method 2: If you need to save and reload
print("\n" + "="*80)
print("Method 2: Save and reload (if needed)")
print("="*80)

# Save to disk
output_dir = "./llm_training_dataset_saved"
ds.save_to_disk(output_dir)
print(f"✅ Saved to {output_dir}")

# Reload from disk - this should work correctly
ds_reloaded = load_dataset(output_dir)
print(f"\nReloaded dataset:")
print(f"Features: {list(ds_reloaded['train'].features.keys())}")
print(f"Number of rows: {len(ds_reloaded['train'])}")
print(f"\nReloaded structure:")
print(ds_reloaded)

# Method 3: Create DatasetDict from JSONL
print("\n" + "="*80)
print("Method 3: Create DatasetDict from JSONL")
print("="*80)

from datasets import DatasetDict

# Load as DatasetDict
ds_dict = load_dataset("json", data_files="./llm_training_dataset.jsonl")
print(f"\nDatasetDict:")
print(ds_dict)
print(f"\nTrain split features: {list(ds_dict['train'].features.keys())}")
print(f"Train split rows: {len(ds_dict['train'])}")


