"""
Upload COT v3 dataset to HuggingFace Hub

This script transforms the CoT reasoning into a separate "thinking" field
in the assistant message, with only the final label as the content.

Format:
{
    "conversations": [
        {"role": "system", "content": "...", "thinking": null},
        {"role": "user", "content": "...", "thinking": null},
        {"role": "assistant", "content": "Clear Reply", "thinking": "First, let me understand..."}
    ]
}
"""

import re
import json
import argparse
from datasets import Dataset
from huggingface_hub import login


def restructure_cot_natural(original_cot: str) -> str:
    """
    Transform step-based CoT format to natural flowing prose.
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
    
    # Clean up extra newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def extract_thinking_and_label(content: str) -> tuple[str, str]:
    """
    Extract the thinking (reasoning) and the final label from assistant content.
    
    Returns:
        tuple: (thinking, label)
    """
    # Find the LABEL line
    label_match = re.search(r'LABEL:\s*(.+?)$', content, re.MULTILINE)
    
    if label_match:
        label = label_match.group(1).strip()
        # Everything before the LABEL line is the thinking
        thinking = content[:label_match.start()].strip()
    else:
        # If no label found, treat the whole thing as thinking
        label = ""
        thinking = content.strip()
    
    # Convert thinking to natural format
    thinking = restructure_cot_natural(thinking)
    
    return thinking, label


def transform_conversation(conversation: list) -> list:
    """
    Transform the conversation to v3 format with separate thinking field.
    """
    new_conversation = []
    
    for message in conversation:
        if message["role"] == "assistant":
            thinking, label = extract_thinking_and_label(message["content"])
            new_conversation.append({
                "role": "assistant",
                "content": label,
                "thinking": thinking
            })
        else:
            # System and user messages get thinking: null
            new_conversation.append({
                "role": message["role"],
                "content": message["content"],
                "thinking": None
            })
    
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
        description="Upload COT v3 dataset to HuggingFace (with thinking field)"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="Frenzyknight/clarity-cot-v3-dataset",
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

    # Transform the conversations to v3 format
    print("Transforming to v3 format (separate thinking field)...")
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
    print("\n" + "=" * 70)
    print("SAMPLE TRANSFORMATION")
    print("=" * 70)
    
    sample = transformed_records[0]["conversations"]
    for msg in sample:
        print(f"\n--- {msg['role'].upper()} ---")
        print(f"content: {msg['content'][:300]}{'...' if len(msg['content']) > 300 else ''}")
        if msg['thinking']:
            print(f"thinking: {msg['thinking'][:400]}{'...' if len(msg['thinking']) > 400 else ''}")
        else:
            print(f"thinking: null")
    
    print("\n" + "=" * 70)

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
