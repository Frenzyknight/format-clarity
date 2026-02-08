"""
Upload COT v2 dataset to HuggingFace Hub

This script transforms the step-based CoT reasoning format into a cleaner
section-header format before uploading.
"""

import re
import json
import argparse
from datasets import Dataset
from huggingface_hub import login


def restructure_cot(original_cot: str) -> str:
    """
    Transform step-based CoT format to section-header format.
    
    Before:
        Step 1 - What the question asks: ...
        Step 2 - How the answer addresses it: ...
        Step 3 - Information type: ...
        Step 4 - Evasion check: ...
        Summary: ...
    
    After:
        **Question core:** ...
        **Response analysis:** ...
        **Evidence:** ...
        **Evasion check:** ...
        **Classification:** ...
    """
    # Remove step markers
    text = re.sub(r'Step \d+ - ', '', original_cot)
    text = re.sub(r'Step \d+:', '', text)
    
    # Add section headers instead
    text = re.sub(r'What the question asks:', '**Question core:**', text)
    text = re.sub(r'How the answer addresses it:', '**Response analysis:**', text)
    text = re.sub(r'Information type:', '**Evidence:**', text)
    text = re.sub(r'Evasion check:', '**Evasion check:**', text)
    text = re.sub(r'Summary:', '**Classification:**', text)
    
    # Clean up extra newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def transform_conversation(conversation: list) -> list:
    """
    Transform the assistant message in a conversation to use the new CoT format.
    """
    new_conversation = []
    for message in conversation:
        if message["role"] == "assistant":
            new_content = restructure_cot(message["content"])
            new_conversation.append({"role": "assistant", "content": new_content})
        else:
            new_conversation.append(message)
    return new_conversation


def load_jsonl(file_path: str) -> list:
    """Load a JSONL file and return a list of records."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Upload restructured COT v2 dataset to HuggingFace"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="Frenzyknight/clarity-cot-v2-dataset",
        help="HuggingFace repo name (e.g., username/dataset-name)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="cot_data/train_cot.jsonl",
        help="Path to the input JSONL file",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private",
    )
    args = parser.parse_args()

    # Login to HuggingFace (will prompt for token if not already logged in)
    print("Logging in to HuggingFace...")
    login()

    # Load the JSONL dataset
    print(f"Loading dataset from {args.data_file}...")
    records = load_jsonl(args.data_file)
    print(f"Loaded {len(records)} examples")

    # Transform the CoT reasoning in each conversation
    print("Transforming CoT reasoning format...")
    transformed_records = []
    for record in records:
        new_record = {
            "conversations": transform_conversation(record["conversations"])
        }
        transformed_records.append(new_record)

    # Create a HuggingFace Dataset
    dataset = Dataset.from_list(transformed_records)

    print(f"Dataset created with {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")
    
    # Show sample of transformation
    print("\n--- Sample transformation ---")
    print("Original assistant message (first 500 chars):")
    original_assistant = records[0]["conversations"][-1]["content"]
    print(original_assistant[:500] + "...")
    print("\nTransformed assistant message (first 500 chars):")
    transformed_assistant = transformed_records[0]["conversations"][-1]["content"]
    print(transformed_assistant[:500] + "...")

    # Push to HuggingFace Hub
    print(f"\nPushing dataset to {args.repo}...")
    dataset.push_to_hub(
        args.repo,
        private=args.private,
    )

    print(f"\nDataset uploaded successfully!")
    print(f"View it at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
