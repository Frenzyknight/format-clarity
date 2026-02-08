"""
Upload Natural CoT dataset to HuggingFace Hub

This script transforms the step-based CoT reasoning format into a natural
flowing prose format before uploading.
"""

import re
import json
import argparse
from datasets import Dataset
from huggingface_hub import login


def restructure_cot_natural(original_cot: str) -> str:
    """
    Transform step-based CoT format to natural flowing prose.
    
    Before:
        Step 1 - What the question asks: The question asks for...
        Step 2 - How the answer addresses it: The answer directly addresses...
        Step 3 - Information type: The key information is explicit...
        Step 4 - Evasion check: There is no meaningful evasion...
        Summary: The question asks... This makes the classification...
        LABEL: Clear Reply
    
    After:
        First, let me understand what the question is asking. The question asks for...

        Now, looking at how the response addresses this. The answer directly addresses...

        Considering the type of information provided: The key information is explicit...

        Let me check for any evasion. There is no meaningful evasion...

        Putting it all together: The question asks... This makes the classification...

        LABEL: Clear Reply
    """
    text = original_cot
    
    # Transform to natural flowing prose
    text = re.sub(
        r'Step 1 - What the question asks:', 
        'First, let me understand what the question is asking.', 
        text
    )
    text = re.sub(
        r'Step 2 - How the answer addresses it:', 
        '\nNow, looking at how the response addresses this.', 
        text
    )
    text = re.sub(
        r'Step 3 - Information type:', 
        '\nConsidering the type of information provided:', 
        text
    )
    text = re.sub(
        r'Step 4 - Evasion check:', 
        '\nLet me check for any evasion.', 
        text
    )
    text = re.sub(
        r'Summary:', 
        '\nPutting it all together:', 
        text
    )
    
    # Clean up extra newlines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def transform_conversation(conversation: list) -> list:
    """
    Transform the assistant message in a conversation to use natural CoT format.
    """
    new_conversation = []
    for message in conversation:
        if message["role"] == "assistant":
            new_content = restructure_cot_natural(message["content"])
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
        description="Upload natural CoT dataset to HuggingFace"
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show transformation preview, don't upload",
    )
    args = parser.parse_args()

    # Load the JSONL dataset
    print(f"Loading dataset from {args.data_file}...")
    records = load_jsonl(args.data_file)
    print(f"Loaded {len(records)} examples")

    # Transform the CoT reasoning in each conversation
    print("Transforming CoT reasoning to natural flowing prose...")
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
    print("\n" + "=" * 60)
    print("SAMPLE TRANSFORMATION")
    print("=" * 60)
    
    print("\n--- ORIGINAL (first 800 chars) ---")
    original_assistant = records[0]["conversations"][-1]["content"]
    print(original_assistant[:800])
    if len(original_assistant) > 800:
        print("...")
    
    print("\n--- TRANSFORMED (first 800 chars) ---")
    transformed_assistant = transformed_records[0]["conversations"][-1]["content"]
    print(transformed_assistant[:800])
    if len(transformed_assistant) > 800:
        print("...")
    
    print("\n" + "=" * 60)

    if args.dry_run:
        print("\nDry run mode - skipping upload.")
        return

    # Login to HuggingFace
    print("\nLogging in to HuggingFace...")
    login()

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
