#!/usr/bin/env python3
"""
Qwen3-32B Difficulty Evaluation Script

Evaluates the Qwen3-32B thinking model on training data to identify "tough" questions
(ones the model gets wrong) and outputs a CSV with difficulty classifications.

Usage:
    python evaluate_difficulty.py [--limit N] [--output PATH]
"""

import argparse
import re
import warnings

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==================== CONFIGURATION ====================
MODEL_NAME = "Qwen/Qwen3-32B"
DATA_PATH = "data/train/train.parquet"
OUTPUT_PATH = "difficulty_analysis.csv"

# Generation parameters for thinking mode
MAX_NEW_TOKENS = 1024  # Longer for thinking mode output
TEMPERATURE = 0.6      # Qwen3 recommended temperature for thinking
TOP_P = 0.95
TOP_K = 20
MAX_INPUT_LENGTH = 4096

# Valid labels for classification
VALID_LABELS = {"Clear Reply", "Clear Non-Reply", "Ambivalent"}

# Suppress attention mask warnings
warnings.filterwarnings("ignore", message=".*attention_mask.*")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-32B on training data to find difficult questions"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of examples to process (for testing)"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_PATH,
        help=f"Output CSV path (default: {OUTPUT_PATH})"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME,
        help=f"Model name/path (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="Disable thinking mode (use standard generation)"
    )
    return parser.parse_args()


def extract_label(response: str) -> str:
    """
    Extract classification label from thinking model response.
    
    Handles both thinking mode output (with <think>...</think> blocks)
    and standard output formats.
    """
    # Remove thinking block if present to get the final answer
    # The thinking block is wrapped in <think>...</think> tags
    response_clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response_clean = response_clean.strip()
    
    # Look for explicit LABEL: pattern (common in COT responses)
    match = re.search(
        r'LABEL:\s*(Clear Reply|Clear Non-Reply|Ambivalent)',
        response_clean,
        re.IGNORECASE
    )
    if match:
        label = match.group(1).strip()
        return normalize_label(label)
    
    # Look for the label as a standalone line or at the end
    # Check for exact label matches (case-insensitive)
    response_lower = response_clean.lower()
    
    # Check last 150 chars for the final answer
    last_part = response_lower[-150:] if len(response_lower) > 150 else response_lower
    
    # Priority order: more specific labels first
    if "clear non-reply" in last_part or "non-reply" in last_part:
        return "Clear Non-Reply"
    elif "clear reply" in last_part:
        return "Clear Reply"
    elif "ambivalent" in last_part:
        return "Ambivalent"
    
    # Check entire response if not found in last part
    if "clear non-reply" in response_lower or "non-reply" in response_lower:
        return "Clear Non-Reply"
    elif "clear reply" in response_lower:
        return "Clear Reply"
    elif "ambivalent" in response_lower:
        return "Ambivalent"
    
    return "PARSE_ERROR"


def normalize_label(label: str) -> str:
    """Normalize label to standard format."""
    label_lower = label.lower().strip()
    if "non-reply" in label_lower or "non reply" in label_lower:
        return "Clear Non-Reply"
    elif "clear reply" in label_lower:
        return "Clear Reply"
    elif "ambivalent" in label_lower:
        return "Ambivalent"
    return label


def extract_ground_truth(conversations: list) -> str:
    """Extract ground truth label from the assistant message in conversations."""
    for msg in conversations:
        if msg.get("role") == "assistant":
            return normalize_label(msg.get("content", "").strip())
    return "UNKNOWN"


def extract_question_snippet(conversations: list, max_len: int = 200) -> str:
    """Extract a snippet of the user's question for reference."""
    for msg in conversations:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Find the "Specific Question" section if present
            match = re.search(r'### Specific Question[^#]*###\s*(.+?)(?:\n|$)', content, re.DOTALL)
            if match:
                snippet = match.group(1).strip()
            else:
                # Fall back to last part of the question
                snippet = content
            
            if len(snippet) > max_len:
                snippet = snippet[:max_len] + "..."
            return snippet
    return ""


def prepare_messages_for_inference(conversations: list, enable_thinking: bool = True) -> list:
    """
    Prepare conversation messages for inference.
    
    Removes the assistant message (ground truth) and optionally enables thinking mode.
    """
    messages = []
    for msg in conversations:
        if msg.get("role") != "assistant":
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Enable thinking mode by appending /think to the user message
    # This is Qwen3's way to activate extended thinking
    if enable_thinking and messages:
        # Find the last user message and append thinking trigger
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                # Qwen3 thinking mode is enabled by default, but we can be explicit
                # The model will use <think>...</think> tags for reasoning
                break
    
    return messages


def load_model(model_name: str):
    """Load model and tokenizer with 2-GPU distribution."""
    print("=" * 60)
    print(f"Loading model: {model_name}")
    print("Distributing across available GPUs with FP16...")
    print("=" * 60)
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\nAvailable GPUs: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    if num_gpus < 2:
        print("\nWarning: Less than 2 GPUs available. Model will use available devices.")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model distributed across GPUs
    print("Loading model with device_map='auto' (distributes across all GPUs)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    print("\nModel loaded successfully!")
    if hasattr(model, 'hf_device_map'):
        devices = set(model.hf_device_map.values())
        print(f"Model distributed across devices: {devices}")
        device_counts = {}
        for layer, device in model.hf_device_map.items():
            device_counts[device] = device_counts.get(device, 0) + 1
        for device, count in sorted(device_counts.items()):
            print(f"  Device {device}: {count} modules")
    
    return model, tokenizer


def load_training_data(data_path: str, limit: int = None) -> pd.DataFrame:
    """Load training data from parquet file."""
    print(f"\nLoading training data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} training examples")
    
    if limit:
        print(f"Limiting to first {limit} examples (for testing)")
        df = df.head(limit)
    
    return df


def run_inference(
    model,
    tokenizer,
    df: pd.DataFrame,
    enable_thinking: bool = True
) -> list:
    """Run inference on all examples and collect results."""
    results = []
    total = len(df)
    
    print("\n" + "=" * 60)
    print(f"Starting inference on {total} examples")
    print(f"Thinking mode: {'enabled' if enable_thinking else 'disabled'}")
    print("=" * 60 + "\n")
    
    for idx, row in tqdm(df.iterrows(), total=total, desc="Evaluating"):
        conversations = row["conversations"]
        
        # Convert numpy array to list if needed
        if hasattr(conversations, 'tolist'):
            conversations = conversations.tolist()
        
        # Extract ground truth before removing assistant message
        golden_label = extract_ground_truth(conversations)
        question_snippet = extract_question_snippet(conversations)
        
        # Prepare messages for inference (remove assistant message)
        messages = prepare_messages_for_inference(conversations, enable_thinking)
        
        # Apply chat template
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            tqdm.write(f"[{idx + 1}/{total}] Template error: {e}")
            results.append({
                "idx": idx,
                "prediction": "ERROR",
                "golden_label": golden_label,
                "difficulty": "unknown",
                "raw_output": f"Template error: {e}",
                "question_snippet": question_snippet,
            })
            continue
        
        # Tokenize
        encoded = tokenizer(prompt, return_tensors="pt")
        inputs = encoded["input_ids"]
        input_len = inputs.shape[-1]
        
        # Skip if too long
        if input_len > MAX_INPUT_LENGTH:
            tqdm.write(f"[{idx + 1}/{total}] SKIPPED (input too long: {input_len} tokens)")
            results.append({
                "idx": idx,
                "prediction": "SKIPPED",
                "golden_label": golden_label,
                "difficulty": "unknown",
                "raw_output": f"Input too long: {input_len} tokens",
                "question_snippet": question_snippet,
            })
            continue
        
        try:
            # Move to model's device
            inputs = inputs.to(model.device)
            attention_mask = torch.ones_like(inputs)
            
            # Generate with thinking-optimized parameters
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode only the new tokens (preserve <think> tags)
            full_response = tokenizer.decode(
                outputs[0, input_len:],
                skip_special_tokens=False
            ).strip()
            
            # Extract prediction label
            prediction = extract_label(full_response)
            
            # Determine difficulty
            if prediction in VALID_LABELS and golden_label in VALID_LABELS:
                difficulty = "easy" if prediction == golden_label else "hard"
            else:
                difficulty = "unknown"
            
            results.append({
                "idx": idx,
                "prediction": prediction,
                "golden_label": golden_label,
                "difficulty": difficulty,
                "raw_output": full_response,
                "question_snippet": question_snippet,
            })
            
        except Exception as e:
            tqdm.write(f"[{idx + 1}/{total}] ERROR: {e}")
            results.append({
                "idx": idx,
                "prediction": "ERROR",
                "golden_label": golden_label,
                "difficulty": "unknown",
                "raw_output": str(e),
                "question_snippet": question_snippet,
            })
        
        # Clear CUDA cache periodically
        if (idx + 1) % 25 == 0:
            torch.cuda.empty_cache()
    
    return results


def print_summary_statistics(results_df: pd.DataFrame):
    """Print summary statistics about difficulty distribution."""
    print("\n" + "=" * 60)
    print("DIFFICULTY ANALYSIS SUMMARY")
    print("=" * 60)
    
    total = len(results_df)
    
    # Count by difficulty
    difficulty_counts = results_df["difficulty"].value_counts()
    
    print(f"\nTotal examples: {total}")
    print("\nDifficulty Distribution:")
    for diff in ["easy", "hard", "unknown"]:
        count = difficulty_counts.get(diff, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {diff:10}: {count:5} ({pct:5.1f}%)")
    
    # Filter to valid predictions only
    valid_mask = results_df["difficulty"].isin(["easy", "hard"])
    valid_df = results_df[valid_mask]
    
    if len(valid_df) > 0:
        # Accuracy
        accuracy = (valid_df["difficulty"] == "easy").sum() / len(valid_df)
        print(f"\nModel Accuracy: {accuracy:.4f}")
        
        # Breakdown by ground truth label
        print("\nDifficulty by Ground Truth Label:")
        for label in VALID_LABELS:
            label_df = valid_df[valid_df["golden_label"] == label]
            if len(label_df) > 0:
                hard_count = (label_df["difficulty"] == "hard").sum()
                hard_pct = hard_count / len(label_df) * 100
                print(f"  {label:20}: {hard_count:4}/{len(label_df):4} hard ({hard_pct:5.1f}%)")
    
    # Parse errors
    parse_errors = (results_df["prediction"] == "PARSE_ERROR").sum()
    skipped = (results_df["prediction"] == "SKIPPED").sum()
    errors = (results_df["prediction"] == "ERROR").sum()
    
    print(f"\nProcessing Issues:")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Skipped:      {skipped}")
    print(f"  Errors:       {errors}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Load training data
    df = load_training_data(DATA_PATH, limit=args.limit)
    
    # Run inference
    enable_thinking = not args.no_thinking
    results = run_inference(model, tokenizer, df, enable_thinking)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = args.output
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")
    
    # Print summary statistics
    print_summary_statistics(results_df)
    
    # Cleanup
    print("\nCleaning up...")
    del model
    torch.cuda.empty_cache()
    print("Done!")
    
    return results_df


if __name__ == "__main__":
    main()
