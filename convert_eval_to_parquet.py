"""
Convert clarity_task_evaluation_dataset.csv to parquet format for HuggingFace upload.
Uses the conversations format with role/content structure.

USAGE:
    python convert_eval_to_parquet.py
"""

import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

# Config
load_dotenv()
INPUT_CSV = "./clarity_task_evaluation_dataset.csv"
OUTPUT_DIR = "./data/eval"
OUTPUT_FILE = "clarity_eval.parquet"


def format_row(row):
    """Format a single row into ChatML conversations format for evaluation"""
    
    # Get the interview question - use 'interview_question' if available, otherwise construct from context
    interview_question = str(row.get('interview_question', '')).strip()
    if not interview_question or interview_question == 'nan':
        interview_question = str(row.get('question', '')).strip()
    
    interview_answer = str(row.get('interview_answer', '')).strip()
    question = str(row.get('question', '')).strip()
    
    # Skip rows with missing essential data
    if not question or question == 'nan':
        return None
    if not interview_answer or interview_answer == 'nan':
        return None
    
    prompt = f"""Based on a segment of the interview where the interviewer asks a series of questions, classify the type of response provided by the interviewee for the following question.

### Classification Categories ###
1. Clear Reply - The information requested is explicitly stated (in the requested form) without having multiple interpretations.
2. Clear Non-Reply - The information requested is not given at all due to ignoring the question itself, need for clarification, or declining to answer
3. Ambivalent - The information requested is given in an incomplete way (e.g., the answer is too general, partial, implicit, dodging, or deflection)

### Full Interview Question ###
{interview_question}

### Full Interview Answer ###
{interview_answer}

### Specific Question to Classify ###
{question}

Classify the response clarity for this specific question. Respond with only one of: Clear Reply, Clear Non-Reply, or Ambivalent"""

    result = {
        "conversations": [
            {
                "role": "system",
                "content": "You are an expert political discourse analyst specializing in classifying response clarity in political interviews. Your task is to determine whether a politician's response to a specific question is a Clear Reply, Clear Non-Reply, or Ambivalent."
            },
            {"role": "user", "content": prompt}
        ]
    }
    
    # If there's a label, include it as assistant response (for training)
    clarity_label = str(row.get('clarity_label', '')).strip()
    if clarity_label and clarity_label != 'nan' and clarity_label != '':
        # Normalize label
        label_map = {
            'clear reply': 'Clear Reply',
            'ambivalent': 'Ambivalent', 
            'clear non-reply': 'Clear Non-Reply',
            'clear non reply': 'Clear Non-Reply',
        }
        normalized_label = label_map.get(clarity_label.lower(), clarity_label)
        result["conversations"].append({
            "role": "assistant",
            "content": normalized_label
        })
    
    return result


def main():
    # Initialize HuggingFace API
    hf_api = HfApi(endpoint="https://huggingface.co", token=os.environ.get("HUGGINGFACE_TOKEN"))
    
    print(f"Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for label column
    if 'clarity_label' in df.columns:
        non_null_labels = df['clarity_label'].notna().sum()
        print(f"Rows with clarity_label: {non_null_labels}")
        if non_null_labels > 0:
            print(f"Label distribution:\n{df['clarity_label'].value_counts()}")
    
    # Format all rows
    formatted = []
    skipped = 0
    for _, row in df.iterrows():
        result = format_row(row)
        if result:
            formatted.append(result)
        else:
            skipped += 1
    
    print(f"\nFormatted rows: {len(formatted)}")
    print(f"Skipped rows (missing data): {skipped}")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame and save as parquet
    formatted_df = pd.DataFrame(formatted)
    output_path = f"{OUTPUT_DIR}/{OUTPUT_FILE}"
    formatted_df.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    # Show sample
    print(f"\n--- Sample conversation ---")
    if len(formatted) > 0:
        sample = formatted[0]
        print(f"System: {sample['conversations'][0]['content'][:100]}...")
        print(f"User prompt length: {len(sample['conversations'][1]['content'])} chars")
        if len(sample['conversations']) > 2:
            print(f"Assistant: {sample['conversations'][2]['content']}")
        else:
            print("(No assistant response - evaluation mode)")
    
    # Upload to HuggingFace
    print(f"\nUploading to HuggingFace...")
    repo_id = "Frenzyknight/clarity-dataset-eval"
    hf_api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    hf_api.upload_folder(folder_path=OUTPUT_DIR, path_in_repo="data/eval", repo_id=repo_id, repo_type="dataset")
    
    print(f"\n✅ Done! Uploaded to: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
