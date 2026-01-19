"""
Format QEvasion dataset from HuggingFace for Unsloth fine-tuning.
Target: 3-class clarity classification (Clear Reply, Ambivalent, Clear Non-Reply)

USAGE:
1. Run: python format_data.py
2. Use the output JSONL files in Unsloth
"""

import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

# config
load_dotenv()
OUTPUT_DIR_TRAIN = "./data/train"
OUTPUT_DIR_TEST = "./data/test"
hf_api = HfApi(endpoint="https://huggingface.co", token=os.environ["HUGGINGFACE_TOKEN"])

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

                Classify the response clarity for this specific question. Respond with only one of: Clear Reply, Clear Non-Reply, or Ambivalent
            """

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
                1. Clear Reply - The information requested is explicitly stated (in the requested form) without having multiple interpretations.
                2. Clear Non-Reply - The information requested is not given at all due to ignoring the question itself, need for clarification, or declining to answer
                3. Ambivalent - The information requested is given in an incomplete way (e.g., the answer is too general, partial, implicit, dodging, or deflection)

                ### Full Interview Question ###
                {str(row['interview_question']).strip()}

                ### Full Interview Answer ###
                {str(row['interview_answer']).strip()}

                ### Specific Question to Classify ###
                {str(row['question']).strip()}

                Classify the response clarity for this specific question. Respond with only one of: Clear Reply, Clear Non-Reply, or Ambivalent
            """

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
        
    # Create output directory
    Path("./data").mkdir(exist_ok=True)
    Path(OUTPUT_DIR_TRAIN).mkdir(exist_ok=True)
    Path(OUTPUT_DIR_TEST).mkdir(exist_ok=True)
    # Format and save training data

    formatted = [format_row(row) for _, row in train_df.iterrows()]
    formatted = pd.DataFrame(formatted)
    train_path = f"{OUTPUT_DIR_TRAIN}/train.parquet"
    formatted.to_parquet(path=train_path, index=False)
    print(f"Saved: {train_path} ({len(formatted)} examples)")
    
    # Format and save test data (for inference)
    test_formatted = [format_test_row(row) for _, row in test_df.iterrows()]
    test_formatted = pd.DataFrame(test_formatted)
    test_path = f"{OUTPUT_DIR_TEST}/test.parquet"
    test_formatted.to_parquet(path=test_path, index=False)
    print(f"Saved: {test_path} ({len(test_formatted)} examples)")
    
    hf_api.create_repo(repo_id="clarity-dataset", repo_type="dataset", exist_ok=True)
    hf_api.upload_folder(folder_path="./data", repo_id="Frenzyknight/clarity-dataset", repo_type="dataset")
    hf_api.upload_folder(folder_path="./data", repo_id="Frenzyknight/clarity-dataset", repo_type="dataset")
    print(f"\n✅ Done!")
    print(f"\nLoad in Unsloth:")
    print(f"   from datasets import load_dataset")
    print(f"   train_dataset = load_dataset('json', data_files='{OUTPUT_DIR_TRAIN}/train.parquet', split='train')")
    print(f"   test_dataset = load_dataset('json', data_files='{OUTPUT_DIR_TEST}/test.parquet', split='train')")


if __name__ == "__main__":
    main()