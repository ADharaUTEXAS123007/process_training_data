"""
LLM Training Dataset Converter
Converts HuggingFace dataset from conversation format to instruction fine-tuning format.

Input format:
[{'role': 'user', 'content': 'What is the HTS US Code for...'},
 {'role': 'assistant', 'content': 'HTS US Code -> 8477.10.9015; 8480.71.8045\nReasoning -> ...'}]

Output format:
{
    'prompt': [{'content': 'instruction template with user question'}],
    'completions': [{'content': 'assistant response'}],
    'answer': 'extracted HTS code(s)',
    'reward': 1.0
}
"""

import pandas as pd
import json
import re
from typing import Dict, List, Any
from pathlib import Path


def extract_hts_codes(text: str) -> str:
    """
    Extract HTS codes from assistant response.
    Looks for patterns like "HTS US Code -> 8477.10.9015; 8480.71.8045"
    or "HTS Code -> 8477.10.9015"
    """
    # Pattern 1: "HTS US Code -> 8477.10.9015; 8480.71.8045"
    pattern1 = r'HTS\s+US\s+Code\s*->\s*([0-9.;\s]+)'
    # Pattern 2: "HTS Code -> 8477.10.9015"
    pattern2 = r'HTS\s+Code\s*->\s*([0-9.;\s]+)'
    # Pattern 3: Just look for HTS codes in format XXXX.XX.XXXX
    pattern3 = r'\b(\d{4}\.\d{2}\.\d{4}(?:\.\d{2})?)\b'
    
    # Try pattern 1 first
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        codes = match.group(1).strip()
        # Clean up: remove extra spaces, keep semicolons
        codes = re.sub(r'\s+', ' ', codes)
        return codes
    
    # Try pattern 2
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        codes = match.group(1).strip()
        codes = re.sub(r'\s+', ' ', codes)
        return codes
    
    # Try pattern 3 - find all HTS codes
    matches = re.findall(pattern3, text)
    if matches:
        return '; '.join(matches)
    
    # If no pattern matches, return empty string
    return ""


def create_prompt_template(user_content: str) -> str:
    """
    Create the prompt template from user content.
    This formats the user question as an instruction.
    """
    # If it already starts with "What is the HTS US Code for", use as is
    if user_content.strip().startswith("What is the HTS US Code for"):
        return user_content.strip()
    
    # Otherwise, format it as a question
    if not user_content.strip().endswith('?'):
        return f"What is the HTS US Code for {user_content.strip()}?"
    
    return user_content.strip()


def convert_conversation_to_training_format(conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Convert a conversation format to instruction fine-tuning format.
    
    Args:
        conversation: List of messages with 'role' and 'content' keys
        
    Returns:
        Dictionary with 'prompt', 'completions', 'answer', and 'reward' keys
    """
    # Extract user and assistant messages
    user_message = None
    assistant_message = None
    
    for msg in conversation:
        if msg.get('role') == 'user':
            user_message = msg.get('content', '')
        elif msg.get('role') == 'assistant':
            assistant_message = msg.get('content', '')
    
    if not user_message or not assistant_message:
        raise ValueError("Missing user or assistant message in conversation")
    
    # Create prompt template
    prompt_text = create_prompt_template(user_message)
    
    # Extract HTS codes
    hts_codes = extract_hts_codes(assistant_message)
    
    # Format as required
    return {
        'prompt': [{'content': prompt_text}],
        'completions': [{'content': assistant_message}],
        'answer': hts_codes,
        'reward': 1.0
    }


def process_dataset(
    dataset_path: str = None,
    output_path: str = "llm_training_dataset.jsonl",
    split: str = "train"
):
    """
    Process the HuggingFace dataset and convert to training format.
    
    Args:
        dataset_path: Path to local dataset file (optional, if None uses HuggingFace)
        output_path: Path to save the converted dataset
        split: Dataset split to use ('train', 'validation', 'test')
    """
    print(f"Loading dataset split: {split}")
    
    # Load dataset from HuggingFace
    splits = {'train': 'train.jsonl', 'validation': 'validation.jsonl', 'test': 'test.jsonl'}
    
    if dataset_path is None:
        # Use HuggingFace dataset
        dataset_url = f"hf://datasets/flexifyai/cross_rulings_hts_dataset_for_tariffs/{splits[split]}"
        print(f"Reading from HuggingFace: {dataset_url}")
        df = pd.read_json(dataset_url, lines=True)
    else:
        # Use local file
        print(f"Reading from local file: {dataset_path}")
        df = pd.read_json(dataset_path, lines=True)
    
    print(f"Loaded {len(df)} rows")
    print(f"\nFirst row sample:")
    print(df.iloc[0])
    print("\n" + "="*80 + "\n")
    
    # Convert each row
    converted_data = []
    errors = []
    
    for idx, row in df.iterrows():
        try:
            # The dataset should have a 'messages' column or similar
            # Check common column names
            if 'messages' in row:
                conversation = row['messages']
            elif 'conversation' in row:
                conversation = row['conversation']
            elif isinstance(row.iloc[0], list):
                # First column is the conversation
                conversation = row.iloc[0]
            else:
                # Try to parse as JSON if it's a string
                if isinstance(row.iloc[0], str):
                    conversation = json.loads(row.iloc[0])
                else:
                    conversation = row.to_dict()
            
            # Convert to training format
            training_format = convert_conversation_to_training_format(conversation)
            converted_data.append(training_format)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} rows...")
                
        except Exception as e:
            error_msg = f"Error processing row {idx}: {str(e)}"
            print(f"⚠️  {error_msg}")
            errors.append({'row': idx, 'error': str(e)})
            continue
    
    print(f"\n✅ Successfully converted {len(converted_data)} rows")
    if errors:
        print(f"⚠️  {len(errors)} rows had errors")
    
    # Save to JSONL format
    output_file = Path(output_path)
    print(f"\nSaving converted dataset to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(converted_data)} rows to {output_file}")
    
    # Display sample of converted data
    if converted_data:
        print("\n" + "="*80)
        print("Sample converted row:")
        print("="*80)
        sample = converted_data[0]
        print(f"Prompt: {sample['prompt']}")
        print(f"\nCompletions (first 200 chars): {sample['completions'][0]['content'][:200]}...")
        print(f"\nAnswer (HTS Codes): {sample['answer']}")
        print(f"\nReward: {sample['reward']}")
        print("="*80)
    
    # Save errors if any
    if errors:
        error_file = output_file.parent / f"{output_file.stem}_errors.json"
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"\n⚠️  Errors saved to: {error_file}")
    
    return converted_data


def main():
    """Main function to run the conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert HuggingFace dataset to instruction fine-tuning format')
    parser.add_argument('--split', type=str, default='train', 
                       choices=['train', 'validation', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--output', type=str, default='llm_training_dataset.jsonl',
                       help='Output file path')
    parser.add_argument('--local-file', type=str, default=None,
                       help='Path to local dataset file (optional, overrides HuggingFace)')
    
    args = parser.parse_args()
    
    try:
        process_dataset(
            dataset_path=args.local_file,
            output_path=args.output,
            split=args.split
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

