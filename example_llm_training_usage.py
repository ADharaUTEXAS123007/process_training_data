"""
Example usage of the LLM Training Dataset Converter

This script demonstrates how to use the converter to process the HuggingFace dataset.
"""

from llm_training_dataset_converter import process_dataset, convert_conversation_to_training_format
import pandas as pd
import json

# Example 1: Process the full dataset from HuggingFace
if __name__ == "__main__":
    print("="*80)
    print("Example 1: Processing HuggingFace Dataset")
    print("="*80)
    
    # Process training split
    try:
        # Login using e.g. `huggingface-cli login` to access this dataset
        splits = {'train': 'train.jsonl', 'validation': 'validation.jsonl', 'test': 'test.jsonl'}
        
        # Read from HuggingFace
        df = pd.read_json(
            "hf://datasets/flexifyai/cross_rulings_hts_dataset_for_tariffs/" + splits["train"], 
            lines=True
        )
        
        print(f"Loaded {len(df)} rows from HuggingFace")
        print(f"\nFirst row structure:")
        print(df.iloc[0])
        print("\n" + "="*80)
        
        # Convert the dataset
        converted_data = process_dataset(
            dataset_path=None,  # Use HuggingFace
            output_path="llm_training_dataset.jsonl",
            split="train"
        )
        
        print(f"\n✅ Conversion complete! Output saved to llm_training_dataset.jsonl")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nNote: Make sure you're logged into HuggingFace:")
        print("  huggingface-cli login")
    
    # Example 2: Convert a single conversation
    print("\n" + "="*80)
    print("Example 2: Converting a Single Conversation")
    print("="*80)
    
    example_conversation = [
        {'role': 'user',
         'content': 'What is the HTS US Code for a horizontal injection molding machine and molds for producing rubber parts?'},
        {'role': 'assistant',
         'content': "HTS US Code -> 8477.10.9015; 8480.71.8045\nReasoning -> The horizontal injection molding machine is classified under HTS code 8477.10.9015, which pertains to machinery for working rubber or plastics or for the manufacture of products from these materials, specifically injection-molding machines used for processing rubber or other thermosetting materials. The machine's features, including a clamping force of 220 tons, powered mold height adjustment, a central ejector, a computerized control system, and a double brush discharging system, align with the definition of machinery classified under this HTS code. \n\nThe molds, specifically designed for injection molding of rubber parts such as O-rings and U-cups, are classified under HTS code 8480.71.8045, which covers molds for rubber or plastics, specifically injection types. The classification is supported by the fact that these molds are used in conjunction with the injection molding machine to produce the specified rubber components. Both classifications carry a duty rate of 3.3 percent ad valorem, confirming their categorization under the relevant HTS codes."}
    ]
    
    try:
        result = convert_conversation_to_training_format(example_conversation)
        print("\nConverted format:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ Error: {e}")

