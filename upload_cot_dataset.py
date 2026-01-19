"""
Upload COT dataset to HuggingFace Hub
"""

from datasets import load_dataset
from huggingface_hub import login
import argparse


def main():
    parser = argparse.ArgumentParser(description="Upload COT dataset to HuggingFace")
    parser.add_argument(
        "--repo",
        type=str,
        default="Frenzyknight/clarity-cot-dataset",
        help="HuggingFace repo name (e.g., username/dataset-name)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="cot_data/train_cot.jsonl",
        help="Path to the JSONL file",
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
    dataset = load_dataset("json", data_files=args.data_file, split="train")

    print(f"Dataset loaded with {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")
    print("\nSample conversation structure:")
    print(f"  - Number of messages: {len(dataset[0]['conversations'])}")
    for msg in dataset[0]["conversations"]:
        print(f"  - {msg['role']}: {msg['content'][:100]}...")

    # Push to HuggingFace Hub
    print(f"\nPushing dataset to {args.repo}...")
    dataset.push_to_hub(
        args.repo,
        private=args.private,
    )

    print("\n✅ Dataset uploaded successfully!")
    print(f"View it at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
