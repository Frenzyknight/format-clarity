"""
OpenAI Judge for Prediction Disagreements

This script compares predictions from two models and uses OpenAI GPT as a judge
to resolve disagreements. The judge can also overrule both models if it has
better reasoning.
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

# HuggingFace dataset config
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

NOTE: Other models predicted "{model1_prediction}" and "{model2_prediction}" but you should make your OWN independent decision based solely on the content above.

Analyze the interview content and classify the response according to the definitions provided."""


def load_eval_from_huggingface() -> pd.DataFrame:
    """Load the QEvasion test dataset from HuggingFace."""
    print(f"  Loading from HuggingFace: {HF_DATASET}")
    eval_df = pd.read_parquet(f"hf://datasets/{HF_DATASET}/{HF_TEST_SPLIT}")
    return eval_df


def load_data(
    pred1_path: str,
    pred2_path: str,
    eval_data_path: str,
    use_huggingface: bool = True
) -> pd.DataFrame:
    """Load and merge prediction files with evaluation dataset."""
    print("\n" + "=" * 60)
    print("Loading data files...")
    print("=" * 60)
    
    # Load prediction files (handle both xlsx and csv)
    if pred1_path.endswith('.xlsx'):
        pred1_df = pd.read_excel(pred1_path)
    else:
        pred1_df = pd.read_csv(pred1_path)
    print(f"  Model 1 predictions: {len(pred1_df)} rows from {pred1_path}")
    
    if pred2_path.endswith('.xlsx'):
        pred2_df = pd.read_excel(pred2_path)
    else:
        pred2_df = pd.read_csv(pred2_path)
    print(f"  Model 2 predictions: {len(pred2_df)} rows from {pred2_path}")
    
    # Load evaluation dataset (from HuggingFace or local file)
    if use_huggingface:
        eval_df = load_eval_from_huggingface()
        print(f"  Evaluation dataset: {len(eval_df)} rows from HuggingFace ({HF_DATASET})")
    else:
        eval_df = pd.read_csv(eval_data_path)
        print(f"  Evaluation dataset: {len(eval_df)} rows from {eval_data_path}")
    
    # Determine the minimum length to align data
    min_len = min(len(pred1_df), len(pred2_df), len(eval_df))
    print(f"\n  Using first {min_len} rows (minimum across all files)")
    
    # Truncate to minimum length
    pred1_df = pred1_df.head(min_len).reset_index(drop=True)
    pred2_df = pred2_df.head(min_len).reset_index(drop=True)
    eval_df = eval_df.head(min_len).reset_index(drop=True)
    
    # Create merged dataframe
    merged_df = pd.DataFrame({
        'index': range(min_len),
        'interview_question': eval_df['interview_question'],
        'interview_answer': eval_df['interview_answer'],
        'question': eval_df['question'],
        'model1_prediction': pred1_df['prediction'],
        'model2_prediction': pred2_df['prediction'],
        'clarity_label': eval_df.get('clarity_label', pred1_df.get('clarity_label', '')),
    })
    
    # Add reasoning columns if available
    if 'full_response' in pred1_df.columns:
        merged_df['model1_reasoning'] = pred1_df['full_response']
    else:
        merged_df['model1_reasoning'] = ''
        
    if 'raw_output' in pred2_df.columns:
        merged_df['model2_reasoning'] = pred2_df['raw_output']
    else:
        merged_df['model2_reasoning'] = ''
    
    return merged_df


def find_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    """Find rows where the two models disagree."""
    # Normalize predictions for comparison
    df['model1_norm'] = df['model1_prediction'].str.strip().str.lower()
    df['model2_norm'] = df['model2_prediction'].str.strip().str.lower()
    
    disagreements = df[df['model1_norm'] != df['model2_norm']].copy()
    
    # Drop temporary columns
    df.drop(columns=['model1_norm', 'model2_norm'], inplace=True)
    disagreements.drop(columns=['model1_norm', 'model2_norm'], inplace=True)
    
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
    
    # Fallback: check last part of response
    last_part = response[-200:].lower()
    if "clear reply" in last_part and "non-reply" not in last_part:
        return "Clear Reply"
    elif "non-reply" in last_part:
        return "Clear Non-Reply"
    elif "ambivalent" in last_part:
        return "Ambivalent"
    
    return "PARSE_ERROR"


def extract_judge_reasoning(response: str) -> str:
    """Extract the reasoning from judge response."""
    match = re.search(r'REASONING:\s*(.*?)(?=FINAL_LABEL:|$)', response, re.IGNORECASE | re.DOTALL)
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
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_retries: int = 3,
) -> dict:
    """Use OpenAI to make an independent judgment on a disagreement."""
    
    # Format the user prompt - only include content, not detailed model reasoning
    user_prompt = JUDGE_USER_TEMPLATE.format(
        interview_question=truncate_text(row['interview_question'], 1000),
        interview_answer=truncate_text(row['interview_answer'], 2000),
        question=truncate_text(row['question'], 500),
        model1_prediction=row['model1_prediction'],
        model2_prediction=row['model2_prediction'],
    )
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
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
                wait_time = 2 ** attempt  # Exponential backoff
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
    model: str = "gpt-4o",
    temperature: float = 0.1,
    rate_limit_delay: float = 0.5,
) -> pd.DataFrame:
    """Run the OpenAI judge on all disagreements."""
    
    print("\n" + "=" * 60)
    print(f"Running OpenAI Judge on {len(disagreements)} disagreements")
    print(f"Model: {model}, Temperature: {temperature}")
    print("=" * 60 + "\n")
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Initialize result columns in main dataframe
    df['judge_prediction'] = 'N/A'
    df['judge_reasoning'] = ''
    df['judge_full_response'] = ''
    df['final_prediction'] = df['model1_prediction']  # Default to model1
    
    # Process disagreements
    for idx, row in tqdm(disagreements.iterrows(), total=len(disagreements), desc="Judging"):
        result = judge_disagreement(client, row, model, temperature)
        
        df.loc[idx, 'judge_prediction'] = result['judge_prediction']
        df.loc[idx, 'judge_reasoning'] = result['judge_reasoning']
        df.loc[idx, 'judge_full_response'] = result['judge_full_response']
        
        # Set final prediction based on judge
        if result['judge_prediction'] not in ['ERROR', 'PARSE_ERROR']:
            df.loc[idx, 'final_prediction'] = result['judge_prediction']
        else:
            # Fallback to model1 on error
            df.loc[idx, 'final_prediction'] = row['model1_prediction']
        
        # Rate limiting
        time.sleep(rate_limit_delay)
    
    return df


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
        
        agreed_with_m1 = sum(judged['judge_prediction'] == judged['model1_prediction'])
        agreed_with_m2 = sum(judged['judge_prediction'] == judged['model2_prediction'])
        overruled_both = num_disagreements - agreed_with_m1 - agreed_with_m2
        errors = sum(judged['judge_prediction'].isin(['ERROR', 'PARSE_ERROR']))
        
        print("\nJudge decisions on disagreements:")
        print(f"  Agreed with Model 1: {agreed_with_m1} ({agreed_with_m1/num_disagreements*100:.1f}%)")
        print(f"  Agreed with Model 2: {agreed_with_m2} ({agreed_with_m2/num_disagreements*100:.1f}%)")
        print(f"  Overruled both: {overruled_both} ({overruled_both/num_disagreements*100:.1f}%)")
        print(f"  Errors: {errors} ({errors/num_disagreements*100:.1f}%)")
    
    # Final prediction distribution
    print("\nFinal prediction distribution:")
    for label in ["Clear Reply", "Clear Non-Reply", "Ambivalent"]:
        count = sum(df['final_prediction'] == label)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI Judge for resolving prediction disagreements"
    )
    parser.add_argument(
        "--pred1",
        type=str,
        default="llama.csv",
        help="Path to first model predictions (xlsx or csv)"
    )
    parser.add_argument(
        "--pred2",
        type=str,
        default="qwen.csv",
        help="Path to second model predictions (xlsx or csv)"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="clarity_task_evaluation_dataset.csv",
        help="Path to local evaluation dataset CSV (only used with --local-eval)"
    )
    parser.add_argument(
        "--local-eval",
        action="store_true",
        help="Use local eval dataset instead of HuggingFace (default: use HuggingFace)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="judge_resolved_predictions.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use as judge (default: gpt-4o)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature (default: 0.1)"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)"
    )
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Load data (default: use HuggingFace for eval dataset)
    use_hf = not args.local_eval
    df = load_data(args.pred1, args.pred2, args.eval_data, use_huggingface=use_hf)
    
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
        df['final_prediction'] = df['model1_prediction']
    else:
        # Run judge on disagreements
        df = run_judge(
            df,
            disagreements,
            model=args.model,
            temperature=args.temperature,
            rate_limit_delay=args.rate_limit,
        )
    
    # Save results
    output_columns = [
        'index',
        'question',
        'model1_prediction',
        'model2_prediction',
        'judge_prediction',
        'judge_reasoning',
        'final_prediction',
        'clarity_label',
    ]
    
    # Only include columns that exist
    output_columns = [col for col in output_columns if col in df.columns]
    
    df[output_columns].to_csv(args.output, index=False)
    
    # Print summary
    print_summary(df, disagreements, args.output)


if __name__ == "__main__":
    main()
