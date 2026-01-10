"""
Format QEvasion dataset from HuggingFace for Unsloth fine-tuning.
Target: 3-class clarity classification (Clear Reply, Ambivalent, Clear Non-Reply)

USAGE:
1. Run: python format_data.py
2. Use the output JSONL files in Unsloth
"""

import pandas as pd
import json
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================
OUTPUT_DIR = "./formatted_data"
VAL_RATIO = 0.1  # 10% for validation

# ============================================

def load_from_huggingface():
    """Load the QEvasion dataset from HuggingFace"""
    splits = {
        'train': 'data/train-00000-of-00001.parquet', 
        'test': 'data/test-00000-of-00001.parquet'
    }
    
    print("Loading dataset from HuggingFace: ailsntua/QEvasion")
    train_df = pd.read_parquet("hf://datasets/ailsntua/QEvasion/" + splits["train"])
    test_df = pd.read_parquet("hf://datasets/ailsntua/QEvasion/" + splits["test"])
    
    return train_df, test_df


def format_row(row):
    """Format a single row into ChatML format for Llama 3.3"""
    
    prompt = f"""Based on a segment of the interview where the interviewer asks a series of questions, classify the type of response provided by the interviewee for the following question.

### Classification Categories ###
1. Clear Reply - The information requested is explicitly stated (in the requested form)
2. Clear Non-Reply - The information requested is not given at all due to ignorance, need for clarification, or declining to answer
3. Ambivalent - The information requested is given in an incomplete way (e.g., the answer is too general, partial, implicit, dodging, or deflection)

### Full Interview Question ###
{str(row['interview_question']).strip()}

### Full Interview Answer ###
{str(row['interview_answer']).strip()}

### Specific Question to Classify ###
{str(row['question']).strip()}

Classify the response clarity for this specific question. Respond with only one of: Clear Reply, Clear Non-Reply, or Ambivalent"""

    # Normalize label
    label = str(row['clarity_label']).strip()
    label_map = {
        'clear reply': 'Clear Reply',
        'ambivalent': 'Ambivalent', 
        'clear non-reply': 'Clear Non-Reply',
        'clear non reply': 'Clear Non-Reply',
    }
    label = label_map.get(label.lower(), label)

    return {
        "conversations": [
            {
                "role": "system",
                "content": "You are an expert political discourse analyst specializing in classifying response clarity in political interviews. Your task is to determine whether a politician's response to a specific question is a Clear Reply, Clear Non-Reply, or Ambivalent."
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label}
        ]
    }


def format_test_row(row):
    """Format a test row (no label) for inference"""
    
    prompt = f"""Based on a segment of the interview where the interviewer asks a series of questions, classify the type of response provided by the interviewee for the following question.

### Classification Categories ###
1. Clear Reply - The information requested is explicitly stated (in the requested form)
2. Clear Non-Reply - The information requested is not given at all due to ignorance, need for clarification, or declining to answer
3. Ambivalent - The information requested is given in an incomplete way (e.g., the answer is too general, partial, implicit, dodging, or deflection)

### Full Interview Question ###
{str(row['interview_question']).strip()}

### Full Interview Answer ###
{str(row['interview_answer']).strip()}

### Specific Question to Classify ###
{str(row['question']).strip()}

Classify the response clarity for this specific question. Respond with only one of: Clear Reply, Clear Non-Reply, or Ambivalent"""

    return {
        "conversations": [
            {
                "role": "system",
                "content": "You are an expert political discourse analyst specializing in classifying response clarity in political interviews. Your task is to determine whether a politician's response to a specific question is a Clear Reply, Clear Non-Reply, or Ambivalent."
            },
            {"role": "user", "content": prompt}
        ]
    }


def main():
    # Load from HuggingFace
    train_df, test_df = load_from_huggingface()
    
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"\nTrain columns: {train_df.columns.tolist()}")
    
    # Show label distribution
    if 'clarity_label' in train_df.columns:
        print(f"\nLabel distribution:\n{train_df['clarity_label'].value_counts()}")
    
    # Clean and shuffle training data
    required = ['interview_question', 'interview_answer', 'question', 'clarity_label']
    train_df = train_df.dropna(subset=required)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train/val
    val_size = int(len(train_df) * VAL_RATIO)
    val_df = train_df[:val_size]
    train_split_df = train_df[val_size:]
    
    print(f"\nAfter split - Train: {len(train_split_df)}, Val: {len(val_df)}")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Format and save training data
    for name, data in [('train', train_split_df), ('val', val_df)]:
        formatted = [format_row(row) for _, row in data.iterrows()]
        path = f"{OUTPUT_DIR}/{name}.jsonl"
        with open(path, 'w') as f:
            for item in formatted:
                f.write(json.dumps(item) + '\n')
        print(f"Saved: {path} ({len(formatted)} examples)")
    
    # Format and save test data (for inference)
    test_formatted = [format_test_row(row) for _, row in test_df.iterrows()]
    test_path = f"{OUTPUT_DIR}/test.jsonl"
    with open(test_path, 'w') as f:
        for item in test_formatted:
            f.write(json.dumps(item) + '\n')
    print(f"Saved: {test_path} ({len(test_formatted)} examples)")

    print(f"\n✅ Done!")
    print(f"\nLoad in Unsloth:")
    print(f"   from datasets import load_dataset")
    print(f"   train_dataset = load_dataset('json', data_files='{OUTPUT_DIR}/train.jsonl', split='train')")
    print(f"   val_dataset = load_dataset('json', data_files='{OUTPUT_DIR}/val.jsonl', split='train')")
    print(f"   test_dataset = load_dataset('json', data_files='{OUTPUT_DIR}/test.jsonl', split='train')")


if __name__ == "__main__":
    main()