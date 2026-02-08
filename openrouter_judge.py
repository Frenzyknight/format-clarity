"""
OpenRouter Judge for Prediction Disagreements

This script compares predictions from Eva4B and Llama models and uses an OpenRouter
model as a judge to resolve disagreements. The judge makes an independent decision
based on the interview content and classification definitions.
"""

import argparse
import os
import re
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# HuggingFace dataset config (fallback)
HF_DATASET = "ailsntua/QEvasion"
HF_TEST_SPLIT = "data/test-00000-of-00001.parquet"

# Judge system prompt - Independent decision making
JUDGE_SYSTEM_PROMPT = """You are an expert political discourse analyst. Your task is to independently classify political interview responses.

### Classification Categories ###
1. **Clear Reply** - The information requested is explicitly stated (in the requested form). The answer directly addresses the question with specific, concrete information.

2. **Clear Non-Reply** - The information requested is not given at all due to:
   - Ignorance ("I don't know")
   - Need for clarification ("What do you mean by...?")
   - Declining to answer ("I can't comment on that")

3. **Ambivalent** - The information requested is given in an incomplete way:
   - Too general (broad statements without specifics)
   - Partial (only part of the question is addressed)
   - Implicit (the answer can be inferred but is not stated directly)
   - Dodging (redirecting to a different topic)
   - Deflection (answering a different question than asked)

### Your Role ###
- Carefully analyze the interview question and answer
- Make your OWN INDEPENDENT judgment based ONLY on the content and definitions above
- Do NOT rely on or be biased by what other models predicted
- Analyze step by step, then provide your final classification
- Be decisive and provide clear justification

### Output Format ###
Provide your response in the following format:
REASONING: [Your step-by-step analysis of the interview content]
FINAL_LABEL: [Clear Reply|Clear Non-Reply|Ambivalent]"""

# User prompt template for judge - focuses on content, not model predictions
JUDGE_USER_TEMPLATE = """Two AI models disagreed on classifying this political interview response. Please make your own independent judgment.

### Interview Question ###
{interview_question}

### Interview Answer ###
{interview_answer}

### Specific Question Being Classified ###
{question}

NOTE: Model 1 (Eva4B) predicted "{eva4b_prediction}" and Model 2 (Llama) predicted "{llama_prediction}" but you should make your OWN independent decision based solely on the content above.

Analyze the interview content and classify the response according to the definitions provided."""


def load_eval_from_huggingface() -> pd.DataFrame:
    """Load the QEvasion test dataset from HuggingFace."""
    print(f"  Loading from HuggingFace: {HF_DATASET}")
    eval_df = pd.read_parquet(f"hf://datasets/{HF_DATASET}/{HF_TEST_SPLIT}")
    return eval_df


def load_data(
    eva4b_path: str,
    llama_path: str,
    eval_data_path: str,
    use_huggingface: bool = False
) -> pd.DataFrame:
    """Load and merge prediction files with evaluation dataset."""
    print("\n" + "=" * 60)
    print("Loading data files...")
    print("=" * 60)
    
    # Load Eva4B predictions
    eva4b_df = pd.read_csv(eva4b_path)
    print(f"  Eva4B predictions: {len(eva4b_df)} rows from {eva4b_path}")
    
    # Load Llama predictions (handle both xlsx and csv)
    if llama_path.endswith('.xlsx'):
        llama_df = pd.read_excel(llama_path)
    else:
        llama_df = pd.read_csv(llama_path)
    print(f"  Llama predictions: {len(llama_df)} rows from {llama_path}")
    
    # Load evaluation dataset (from HuggingFace or local file)
    if use_huggingface:
        eval_df = load_eval_from_huggingface()
        print(f"  Evaluation dataset: {len(eval_df)} rows from HuggingFace ({HF_DATASET})")
    else:
        eval_df = pd.read_csv(eval_data_path)
        print(f"  Evaluation dataset: {len(eval_df)} rows from {eval_data_path}")
    
    # Determine the minimum length to align data
    min_len = min(len(eva4b_df), len(llama_df), len(eval_df))
    print(f"\n  Using first {min_len} rows (minimum across all files)")
    
    # Truncate to minimum length
    eva4b_df = eva4b_df.head(min_len).reset_index(drop=True)
    llama_df = llama_df.head(min_len).reset_index(drop=True)
    eval_df = eval_df.head(min_len).reset_index(drop=True)
    
    # Create merged dataframe
    merged_df = pd.DataFrame({
        'index': range(min_len),
        'interview_question': eval_df['interview_question'],
        'interview_answer': eval_df['interview_answer'],
        'question': eval_df['question'],
        'eva4b_prediction': eva4b_df['prediction'],
        'llama_prediction': llama_df['prediction'],
    })
    
    # Add clarity_label from eval_df or llama_df if available
    if 'clarity_label' in eval_df.columns:
        merged_df['clarity_label'] = eval_df['clarity_label']
    elif 'clarity_label' in llama_df.columns:
        merged_df['clarity_label'] = llama_df['clarity_label']
    else:
        merged_df['clarity_label'] = ''
    
    # Add raw output/reasoning columns if available
    if 'raw_output' in eva4b_df.columns:
        merged_df['eva4b_reasoning'] = eva4b_df['raw_output']
    else:
        merged_df['eva4b_reasoning'] = ''
        
    if 'full_response' in llama_df.columns:
        merged_df['llama_reasoning'] = llama_df['full_response']
    else:
        merged_df['llama_reasoning'] = ''
    
    return merged_df


def find_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    """Find rows where the two models disagree."""
    # Normalize predictions for comparison
    df['eva4b_norm'] = df['eva4b_prediction'].str.strip().str.lower()
    df['llama_norm'] = df['llama_prediction'].str.strip().str.lower()
    
    disagreements = df[df['eva4b_norm'] != df['llama_norm']].copy()
    
    # Drop temporary columns
    df.drop(columns=['eva4b_norm', 'llama_norm'], inplace=True)
    disagreements.drop(columns=['eva4b_norm', 'llama_norm'], inplace=True)
    
    return disagreements


def extract_judge_label(response: str) -> str:
    """Extract the final label from judge response."""
    # Try to find FINAL_LABEL pattern
    match = re.search(r'FINAL_LABEL:\s*(Clear Reply|Clear Non-Reply|Ambivalent)', response, re.IGNORECASE)
    if match:
        label_map = {
            "clear reply": "Clear Reply",
            "clear non-reply": "Clear Non-Reply",
            "ambivalent": "Ambivalent"
        }
        return label_map.get(match.group(1).lower(), match.group(1))
    
    # Try to find LABEL pattern
    match = re.search(r'LABEL:\s*(Clear Reply|Clear Non-Reply|Ambivalent)', response, re.IGNORECASE)
    if match:
        label_map = {
            "clear reply": "Clear Reply",
            "clear non-reply": "Clear Non-Reply",
            "ambivalent": "Ambivalent"
        }
        return label_map.get(match.group(1).lower(), match.group(1))
    
    # Fallback: check last part of response
    last_part = response[-300:].lower()
    if "clear reply" in last_part and "non-reply" not in last_part:
        return "Clear Reply"
    elif "non-reply" in last_part:
        return "Clear Non-Reply"
    elif "ambivalent" in last_part:
        return "Ambivalent"
    
    return "PARSE_ERROR"


def extract_judge_reasoning(response: str) -> str:
    """Extract the reasoning from judge response."""
    match = re.search(r'REASONING:\s*(.*?)(?=FINAL_LABEL:|LABEL:|$)', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return response


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate text to maximum characters."""
    if pd.isna(text):
        return ""
    text = str(text)
    if len(text) > max_chars:
        return text[:max_chars] + "... [truncated]"
    return text


def judge_disagreement(
    client: OpenAI,
    row: pd.Series,
    model: str = "anthropic/claude-3.5-sonnet",
    temperature: float = 0.1,
    max_retries: int = 3,
    use_reasoning: bool = False,
) -> dict:
    """Use OpenRouter to make an independent judgment on a disagreement."""
    
    # Format the user prompt - only include content, not detailed model reasoning
    user_prompt = JUDGE_USER_TEMPLATE.format(
        interview_question=truncate_text(row['interview_question'], 1500),
        interview_answer=truncate_text(row['interview_answer'], 2500),
        question=truncate_text(row['question'], 500),
        eva4b_prediction=row['eva4b_prediction'],
        llama_prediction=row['llama_prediction'],
    )
    
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    # Optional reasoning mode
    extra_body = {}
    if use_reasoning:
        extra_body["reasoning"] = {"enabled": True}
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                extra_body=extra_body if extra_body else None,
            )
            
            response_text = response.choices[0].message.content
            judge_label = extract_judge_label(response_text)
            judge_reasoning = extract_judge_reasoning(response_text)
            
            return {
                "judge_prediction": judge_label,
                "judge_reasoning": judge_reasoning,
                "judge_full_response": response_text,
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # Exponential backoff
                print(f"    API error (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return {
                    "judge_prediction": "ERROR",
                    "judge_reasoning": str(e),
                    "judge_full_response": str(e),
                }


def run_judge(
    df: pd.DataFrame,
    disagreements: pd.DataFrame,
    model: str = "anthropic/claude-3.5-sonnet",
    temperature: float = 0.1,
    rate_limit_delay: float = 1.0,
    use_reasoning: bool = False,
    checkpoint_every: int = 25,
    output_path: str = "openrouter_judge_results.csv",
) -> pd.DataFrame:
    """Run the OpenRouter judge on all disagreements."""
    
    print("\n" + "=" * 60)
    print(f"Running OpenRouter Judge on {len(disagreements)} disagreements")
    print(f"Model: {model}, Temperature: {temperature}")
    print(f"Reasoning mode: {'enabled' if use_reasoning else 'disabled'}")
    print("=" * 60 + "\n")
    
    # Get OpenRouter API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    # Initialize OpenRouter client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    # Initialize result columns in main dataframe
    df['judge_prediction'] = 'N/A'
    df['judge_reasoning'] = ''
    df['judge_full_response'] = ''
    df['final_prediction'] = df['eva4b_prediction']  # Default to Eva4B
    
    # For agreements, set final prediction based on both models agreeing
    agreements_mask = df['judge_prediction'] == 'N/A'
    
    # Process disagreements
    processed = 0
    for idx, row in tqdm(disagreements.iterrows(), total=len(disagreements), desc="Judging"):
        result = judge_disagreement(
            client, row, model, temperature, 
            use_reasoning=use_reasoning
        )
        
        df.loc[idx, 'judge_prediction'] = result['judge_prediction']
        df.loc[idx, 'judge_reasoning'] = result['judge_reasoning']
        df.loc[idx, 'judge_full_response'] = result['judge_full_response']
        
        # Set final prediction based on judge
        if result['judge_prediction'] not in ['ERROR', 'PARSE_ERROR']:
            df.loc[idx, 'final_prediction'] = result['judge_prediction']
        else:
            # Fallback to Eva4B on error
            df.loc[idx, 'final_prediction'] = row['eva4b_prediction']
        
        processed += 1
        
        # Checkpoint: save intermediate results
        if processed % checkpoint_every == 0:
            checkpoint_path = output_path.replace('.csv', '_checkpoint.csv')
            save_columns = [
                'index', 'question', 'eva4b_prediction', 'llama_prediction',
                'judge_prediction', 'judge_reasoning', 'final_prediction', 'clarity_label'
            ]
            save_columns = [col for col in save_columns if col in df.columns]
            df[save_columns].to_csv(checkpoint_path, index=False)
            print(f"\n  Checkpoint saved at {processed} disagreements")
        
        # Rate limiting
        time.sleep(rate_limit_delay)
    
    return df


def compute_accuracy(df: pd.DataFrame, pred_col: str, label_col: str = 'clarity_label') -> float:
    """Compute accuracy between predictions and ground truth labels."""
    if label_col not in df.columns or df[label_col].isna().all() or (df[label_col] == '').all():
        return None
    
    # Filter valid rows
    valid_mask = (
        df[label_col].notna() & 
        (df[label_col] != '') & 
        df[pred_col].notna() & 
        (df[pred_col] != '') &
        ~df[pred_col].isin(['ERROR', 'PARSE_ERROR', 'N/A'])
    )
    
    if valid_mask.sum() == 0:
        return None
    
    correct = (df.loc[valid_mask, pred_col].str.lower().str.strip() == 
               df.loc[valid_mask, label_col].str.lower().str.strip()).sum()
    
    return correct / valid_mask.sum()


def print_summary(df: pd.DataFrame, disagreements: pd.DataFrame, output_file: str):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("JUDGE EVALUATION COMPLETE")
    print("=" * 60 + "\n")
    
    total = len(df)
    num_disagreements = len(disagreements)
    num_agreements = total - num_disagreements
    
    print(f"Total examples: {total}")
    print(f"  Agreements (no judge needed): {num_agreements} ({num_agreements/total*100:.1f}%)")
    print(f"  Disagreements (judged): {num_disagreements} ({num_disagreements/total*100:.1f}%)")
    
    if num_disagreements > 0:
        # Judge decision breakdown
        judged = df[df['judge_prediction'] != 'N/A']
        
        agreed_with_eva4b = sum(judged['judge_prediction'] == judged['eva4b_prediction'])
        agreed_with_llama = sum(judged['judge_prediction'] == judged['llama_prediction'])
        overruled_both = num_disagreements - agreed_with_eva4b - agreed_with_llama
        errors = sum(judged['judge_prediction'].isin(['ERROR', 'PARSE_ERROR']))
        
        print("\nJudge decisions on disagreements:")
        print(f"  Agreed with Eva4B: {agreed_with_eva4b} ({agreed_with_eva4b/num_disagreements*100:.1f}%)")
        print(f"  Agreed with Llama: {agreed_with_llama} ({agreed_with_llama/num_disagreements*100:.1f}%)")
        print(f"  Overruled both: {overruled_both} ({overruled_both/num_disagreements*100:.1f}%)")
        print(f"  Errors: {errors} ({errors/num_disagreements*100:.1f}%)")
    
    # Final prediction distribution
    print("\nFinal prediction distribution:")
    for label in ["Clear Reply", "Clear Non-Reply", "Ambivalent"]:
        count = sum(df['final_prediction'] == label)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Accuracy metrics if ground truth available
    print("\n" + "-" * 50)
    print("ACCURACY COMPARISON (vs Ground Truth)")
    print("-" * 50)
    
    eva4b_acc = compute_accuracy(df, 'eva4b_prediction')
    llama_acc = compute_accuracy(df, 'llama_prediction')
    final_acc = compute_accuracy(df, 'final_prediction')
    
    if eva4b_acc is not None:
        print(f"\n  Eva4B Accuracy:   {eva4b_acc*100:.2f}%")
    else:
        print("\n  Eva4B Accuracy:   N/A (no ground truth)")
        
    if llama_acc is not None:
        print(f"  Llama Accuracy:   {llama_acc*100:.2f}%")
    else:
        print("  Llama Accuracy:   N/A (no ground truth)")
        
    if final_acc is not None:
        print(f"  Final Accuracy:   {final_acc*100:.2f}%")
        
        if eva4b_acc and llama_acc:
            best_single = max(eva4b_acc, llama_acc)
            improvement = final_acc - best_single
            print(f"\n  Improvement over best single model: {improvement*100:+.2f}%")
    else:
        print("  Final Accuracy:   N/A (no ground truth)")
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="OpenRouter Judge for resolving Eva4B vs Llama prediction disagreements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default model (Claude 3.5 Sonnet)
  python openrouter_judge.py

  # Use a different judge model
  python openrouter_judge.py --model google/gemini-2.0-flash-exp:free

  # Use reasoning mode (extended thinking)
  python openrouter_judge.py --reasoning

  # Custom prediction files
  python openrouter_judge.py --eva4b my_eva4b.csv --llama my_llama.csv

  # Test with limited examples
  python openrouter_judge.py --limit 50

Popular OpenRouter Models:
  anthropic/claude-3.5-sonnet       - Claude 3.5 Sonnet (balanced)
  anthropic/claude-3-opus           - Claude 3 Opus (most capable)
  google/gemini-2.0-flash-exp:free  - Gemini 2.0 Flash (free tier)
  openai/gpt-4o                     - GPT-4o (latest)
  meta-llama/llama-3.3-70b-instruct - Llama 3.3 70B
  qwen/qwen-2.5-72b-instruct        - Qwen 2.5 72B
        """
    )
    parser.add_argument(
        "--eva4b",
        type=str,
        default="clarity_eval_predictions_noncot.csv",
        help="Path to Eva4B predictions CSV (default: clarity_eval_predictions_noncot.csv)"
    )
    parser.add_argument(
        "--llama",
        type=str,
        default="llama.csv",
        help="Path to Llama predictions (xlsx or csv, default: llama.csv)"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="clarity_task_evaluation_dataset.csv",
        help="Path to evaluation dataset CSV with interview Q&A"
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Use HuggingFace dataset instead of local eval CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="openrouter_judge_results.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-3.5-sonnet",
        help="OpenRouter model to use as judge (default: anthropic/claude-3.5-sonnet)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature (default: 0.1)"
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable reasoning mode (extended thinking)"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save checkpoint every N disagreements (default: 25)"
    )
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    # Load data
    df = load_data(
        args.eva4b, 
        args.llama, 
        args.eval_data, 
        use_huggingface=args.use_huggingface
    )
    
    if args.limit:
        print(f"\nLimiting to first {args.limit} examples")
        df = df.head(args.limit)
    
    # Find disagreements
    disagreements = find_disagreements(df)
    print(f"\nFound {len(disagreements)} disagreements out of {len(df)} total ({len(disagreements)/len(df)*100:.1f}%)")
    
    if len(disagreements) == 0:
        print("\nNo disagreements found! Both models agree on all predictions.")
        df['judge_prediction'] = 'N/A'
        df['judge_reasoning'] = ''
        df['final_prediction'] = df['eva4b_prediction']
    else:
        # Run judge on disagreements
        df = run_judge(
            df,
            disagreements,
            model=args.model,
            temperature=args.temperature,
            rate_limit_delay=args.rate_limit,
            use_reasoning=args.reasoning,
            checkpoint_every=args.checkpoint_every,
            output_path=args.output,
        )
    
    # Save results
    output_columns = [
        'index',
        'question',
        'eva4b_prediction',
        'llama_prediction',
        'judge_prediction',
        'judge_reasoning',
        'final_prediction',
        'clarity_label',
    ]
    
    # Only include columns that exist
    output_columns = [col for col in output_columns if col in df.columns]
    
    df[output_columns].to_csv(args.output, index=False)
    
    # Save detailed results with full judge responses
    detailed_output = args.output.replace('.csv', '_detailed.csv')
    df.to_csv(detailed_output, index=False)
    print(f"\nDetailed results saved to: {detailed_output}")
    
    # Clean up checkpoint file if it exists
    checkpoint_path = args.output.replace('.csv', '_checkpoint.csv')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed checkpoint file: {checkpoint_path}")
    
    # Print summary
    print_summary(df, disagreements, args.output)


if __name__ == "__main__":
    main()
