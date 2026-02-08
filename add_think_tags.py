"""
Transform existing COT training data to add <think></think> tags around reasoning.

Usage:
    python add_think_tags.py

This will:
1. Read cot_data/train_cot.jsonl
2. Wrap the reasoning (before LABEL:) in <think></think> tags
3. Save to cot_data/train_cot_think.jsonl
4. Optionally update data/train/train.parquet
"""

import json
import re
from pathlib import Path

INPUT_FILE = "cot_data/train_cot.jsonl"
OUTPUT_FILE = "cot_data/train_cot_think.jsonl"
PARQUET_OUTPUT = "data/train/train.parquet"


def add_think_tags(content: str) -> str:
    """Add <think></think> tags around reasoning in assistant content."""
    
    # Check if already has think tags
    if "<think>" in content:
        return content
    
    # Split at LABEL: to separate reasoning from final answer
    match = re.search(r'(.*?)(LABEL:\s*.+)$', content, re.DOTALL)
    
    if match:
        reasoning = match.group(1).strip()
        label_part = match.group(2).strip()
        return f"<think>\n{reasoning}\n</think>\n\n{label_part}"
    
    # If no LABEL found, just wrap everything
    return f"<think>\n{content.strip()}\n</think>"


def transform_jsonl():
    """Transform JSONL file to add think tags."""
    
    if not Path(INPUT_FILE).exists():
        print(f"❌ Input file not found: {INPUT_FILE}")
        return
    
    transformed = []
    
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            
            # Find and transform assistant content
            for msg in item.get("conversations", []):
                if msg.get("role") == "assistant":
                    msg["content"] = add_think_tags(msg["content"])
            
            transformed.append(item)
    
    # Save transformed data
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        for item in transformed:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Transformed {len(transformed)} examples")
    print(f"   Output: {OUTPUT_FILE}")
    
    # Also save as parquet for training
    try:
        import pandas as pd
        
        # Convert to DataFrame format expected by training
        df = pd.DataFrame({"conversations": [item["conversations"] for item in transformed]})
        
        Path(PARQUET_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PARQUET_OUTPUT, index=False)
        
        print(f"✅ Updated parquet: {PARQUET_OUTPUT}")
    except ImportError:
        print("⚠️ pandas not available, skipping parquet output")
    
    # Show example
    if transformed:
        print("\n📝 Example transformed output:")
        example = transformed[0]["conversations"][-1]["content"]
        print(example[:500] + "..." if len(example) > 500 else example)


if __name__ == "__main__":
    transform_jsonl()
