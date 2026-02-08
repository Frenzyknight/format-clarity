"""
Evaluate Clarity Model on Evaluation Dataset
Loads the full model across 2 GPUs and runs inference
"""

import argparse
import pandas as pd
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
# System prompt (matching COT training format)
SYSTEM_PROMPT = "You are an expert political discourse analyst. Analyze political interviews step by step and classify response clarity."

# User prompt template (matching COT training format)
USER_TEMPLATE = """Based on a segment of the interview where the interviewer asks a series of questions, classify the type of response provided by the interviewee for the following question.

### Classification Categories ###
1. Clear Reply - The information requested is explicitly stated (in the requested form)
2. Clear Non-Reply - The information requested is not given at all due to ignorance, need for clarification, or declining to answer
3. Ambivalent - The information requested is given in an incomplete way (e.g., the answer is too general, partial, implicit, dodging, or deflection)

### Full Interview Question ###
{interview_question}

### Full Interview Answer ###
{interview_answer}

### Specific Question to Classify ###
{question}

Think step by step, then provide your classification."""


def extract_label(response: str) -> str:
    """Extract the final label from COT response."""
    match = re.search(r'LABEL:\s*(Clear Reply|Clear Non-Reply|Ambivalent)', response, re.IGNORECASE)
    if match:
        label_map = {
            "clear reply": "Clear Reply",
            "clear non-reply": "Clear Non-Reply",
            "ambivalent": "Ambivalent"
        }
        return label_map.get(match.group(1).lower(), match.group(1))
    
    # Fallback: check last 100 chars
    last_part = response[-100:].lower()
    if "clear reply" in last_part and "non-reply" not in last_part:
        return "Clear Reply"
    elif "non-reply" in last_part:
        return "Clear Non-Reply"
    elif "ambivalent" in last_part:
        return "Ambivalent"
    
    return "PARSE_ERROR"


def load_model(model_name: str, num_gpus: int = 2):
    """Load model distributed across multiple GPUs."""
    print("\n" + "=" * 60)
    print(f"Loading model: {model_name}")
    print(f"Distributing across {num_gpus} GPUs")
    print("=" * 60 + "\n")
    
    # Check available GPUs
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    
    if available_gpus < num_gpus:
        print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} available")
        num_gpus = available_gpus
    
    for i in range(available_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model with auto device mapping...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically distribute across GPUs
        torch_dtype=torch.float16,  # Use FP16 for efficiency
        trust_remote_code=True,
    )
    
    print("\n✅ Model loaded successfully!")
    print(f"Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}")
    
    return model, tokenizer


def format_messages(row: pd.Series) -> list:
    """Format a row into conversation messages."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            interview_question=str(row["interview_question"]),
            interview_answer=str(row["interview_answer"]),
            question=str(row["question"])
        )}
    ]


def run_inference(
    model,
    tokenizer,
    eval_df: pd.DataFrame,
    max_new_tokens: int = 600,
    temperature: float = 0.1,
    max_input_length: int = 3500,
) -> list:
    """Run inference on the evaluation dataset."""
    
    results = []
    total = len(eval_df)
    
    print("\n" + "=" * 60)
    print(f"Starting inference on {total} examples")
    print("=" * 60 + "\n")
    
    # Use tqdm for progress bar
    for idx, row in tqdm(eval_df.iterrows(), total=total, desc="Evaluating"):
        # Format messages
        messages = format_messages(row)
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        
        input_len = inputs.shape[-1]
        
        # Skip if too long
        if input_len > max_input_length:
            tqdm.write(f"[{idx + 1}/{total}] SKIPPED (input too long: {input_len} tokens)")
            results.append({
                "index": row.get("index", idx),
                "question": row["question"],
                "prediction": "SKIPPED",
                "full_response": f"Input too long: {input_len} tokens"
            })
            continue
        
        try:
            # Move to first GPU (model handles distribution)
            inputs = inputs.to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode response
            full_response = tokenizer.decode(
                outputs[0, input_len:],
                skip_special_tokens=True
            ).strip()
            
            # Extract label
            prediction = extract_label(full_response)
            
            results.append({
                "index": row.get("index", idx),
                "question": row["question"],
                "prediction": prediction,
                "full_response": full_response
            })
            
        except Exception as e:
            tqdm.write(f"[{idx + 1}/{total}] ERROR: {e}")
            results.append({
                "index": row.get("index", idx),
                "question": row["question"],
                "prediction": "ERROR",
                "full_response": str(e)
            })
        
        # Clear cache periodically
        if (idx + 1) % 25 == 0:
            torch.cuda.empty_cache()
    
    return results


def print_summary(results: list, output_file: str):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60 + "\n")
    
    # Count categories
    parse_errors = sum(1 for r in results if r['prediction'] == 'PARSE_ERROR')
    skipped = sum(1 for r in results if r['prediction'] == 'SKIPPED')
    errors = sum(1 for r in results if r['prediction'] == 'ERROR')
    successful = len(results) - parse_errors - skipped - errors
    
    print(f"✅ Saved {len(results)} predictions to {output_file}")
    print("\n📊 Summary:")
    print(f"   Total:         {len(results)}")
    print(f"   Successful:    {successful}")
    print(f"   Parse errors:  {parse_errors}")
    print(f"   Skipped:       {skipped}")
    print(f"   Errors:        {errors}")
    
    # Prediction distribution
    print("\n📈 Prediction distribution:")
    for label in ["Clear Reply", "Clear Non-Reply", "Ambivalent"]:
        count = sum(1 for r in results if r['prediction'] == label)
        pct = count / len(results) * 100 if results else 0
        print(f"   {label}: {count} ({pct:.1f}%)")
    
    # Sample predictions
    print("\n🔍 Sample predictions (first 5):")
    for i, r in enumerate(results[:5]):
        q_preview = r['question'][:50] + "..." if len(r['question']) > 50 else r['question']
        print(f"   {i+1}. {r['prediction']}: {q_preview}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Clarity Model")
    parser.add_argument(
        "--model",
        type=str,
        default="Frenzyknight/Clarity-llama-70b",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="clarity_task_evaluation_dataset.csv",
        help="Path to evaluation CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="clarity_eval_predictions.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=600,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for testing)"
    )
    args = parser.parse_args()
    
    # Load data
    print(f"\n📂 Loading evaluation data from: {args.data}")
    eval_df = pd.read_csv(args.data)
    print(f"   Loaded {len(eval_df)} examples")
    
    if args.limit:
        print(f"   Limiting to first {args.limit} examples")
        eval_df = eval_df.head(args.limit)
    
    # Load model
    model, tokenizer = load_model(args.model, args.num_gpus)
    
    # Run inference
    results = run_inference(
        model,
        tokenizer,
        eval_df,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    
    # Print summary
    print_summary(results, args.output)


if __name__ == "__main__":
    main()
