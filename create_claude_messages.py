"""
Generate messages JSON for the first 50 Q/A pairs from Frenzyknight/clarity-dataset test split
Format: Array of messages with role (system/user) for Claude API
"""

import json
from datasets import load_dataset

def main():
    # Load dataset from HuggingFace
    print("Loading dataset from HuggingFace: Frenzyknight/clarity-dataset")
    ds = load_dataset("Frenzyknight/clarity-dataset", split="test")
    print(f"Total test examples: {len(ds)}")
    
    # Take first 50
    ds = ds.select(range(min(50, len(ds))))
    print(f"Using first {len(ds)} examples")
    
    # Build messages array - concatenate all conversations
    all_messages = []
    
    for i, example in enumerate(ds):
        conversations = example['conversations']
        for msg in conversations:
            all_messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
    
    # Save to file
    output_path = "claude_prompt_50_messages.json"
    with open(output_path, "w") as f:
        json.dump(all_messages, f, indent=2)
    
    print(f"\nMessages saved to: {output_path}")
    print(f"Total messages: {len(all_messages)}")
    print(f"Structure: {len(ds)} examples x 2 messages (system + user) = {len(all_messages)} messages")

if __name__ == "__main__":
    main()
