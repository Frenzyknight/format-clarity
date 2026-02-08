"""
Predict on HuggingFace QEvasion (Clarity) Dataset using OpenRouter API

Uses Arcee Trinity model with reasoning enabled via OpenRouter.
"""

import argparse
import os
import re
import time
import traceback

import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# HuggingFace dataset config
HF_DATASET = "ailsntua/QEvasion"
HF_TEST_SPLIT = "data/test-00000-of-00001.parquet"

# System prompt for classification
SYSTEM_PROMPT = """You are an expert political discourse analyst. Analyze political interviews step by step and classify response clarity."""

# User prompt template
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

# Few-shot examples in conversation format (matching training data)
FEW_SHOT_EXAMPLES = [
    # Example 1: Clear Reply
    {
        "role": "user",
        "content": """Based on a segment of the interview where the interviewer asks a series of questions, classify the type of response provided by the interviewee for the following question.

### Classification Categories ###
1. Clear Reply - The information requested is explicitly stated (in the requested form)
2. Clear Non-Reply - The information requested is not given at all due to ignorance, need for clarification, or declining to answer
3. Ambivalent - The information requested is given in an incomplete way (e.g., the answer is too general, partial, implicit, dodging, or deflection)

### Full Interview Question ###
Q. Of the Biden administration. And accused the United States of containing China while pushing for diplomatic talks. How would you respond to that?

### Full Interview Answer ###
Well, look, first of all, I am sincere about getting the relationship right. And so, really, what this trip was about—it was less about containing China. I don't want to contain China. I just want to make sure that we have a relationship with China that is on the up and up. It's not about containing China; it's about having a stable base in the Indo-Pacific. It's not about isolating China. It's about making sure the rules of the road are abided by.

### Specific Question to Classify ###
How would you respond to the accusation that the United States is containing China while pushing for diplomatic talks?

Think step by step, then provide your classification."""
    },
    {
        "role": "assistant",
        "content": """Step 1 - What the question asks: The question asks for the respondent's substantive response to a specific accusation: that the United States is trying to contain China while simultaneously pushing for diplomatic talks.

Step 2 - How the answer addresses it: The answer directly addresses this by explicitly rejecting the premise that U.S. policy is about containing China. The respondent says, "it was less about containing China. I don't want to contain China," and explains that the goal is to have a stable, rules-based relationship.

Step 3 - Information type: The key information is explicit. The respondent explicitly states: (1) "I don't want to contain China," (2) "It's not about containing China; it's about having a stable base in the Indo-Pacific," and (3) "It's not about isolating China."

Step 4 - Evasion check: There is no meaningful evasion on the core point. The central accusation is confronted head-on and explicitly rejected multiple times.

Summary: The accusation is directly confronted and explicitly rejected, and the policy goal is clearly redefined. The requested information is fully and clearly provided.

LABEL: Clear Reply"""
    },
    # Example 2: Ambivalent
    {
        "role": "user",
        "content": """Based on a segment of the interview where the interviewer asks a series of questions, classify the type of response provided by the interviewee for the following question.

### Classification Categories ###
1. Clear Reply - The information requested is explicitly stated (in the requested form)
2. Clear Non-Reply - The information requested is not given at all due to ignorance, need for clarification, or declining to answer
3. Ambivalent - The information requested is given in an incomplete way (e.g., the answer is too general, partial, implicit, dodging, or deflection)

### Full Interview Question ###
Q. Do you think President Xi is being sincere about getting the relationship back on track as he bans Apple in China?

### Full Interview Answer ###
Well, look, first of all, I am sincere about getting the relationship right. I hope that—I think that Xi has some difficulties right now. All countries end up with difficulties, and he had some economic difficulties he's working his way through. I want to see China succeed economically, but I want to see them succeed by the rules.

### Specific Question to Classify ###
Do you think President Xi is being sincere about getting the relationship back on track as he bans Apple in China?

Think step by step, then provide your classification."""
    },
    {
        "role": "assistant",
        "content": """Step 1 - What the question asks: The question asks for a clear judgment about President Xi's sincerity: Does the interviewee think Xi is genuinely trying to get the U.S.–China relationship back on track, especially in light of actions like banning Apple in China?

Step 2 - How the answer addresses it: The answer touches on Xi and the relationship but does not give a clear, direct yes-or-no assessment of Xi's sincerity. The speaker says, "I am sincere about getting the relationship right," then discusses Xi's difficulties, but never explicitly states whether Xi is sincere.

Step 3 - Information type: The relevant information is implicit and partial. The speaker implies a desire for a constructive relationship but does not explicitly answer the sincerity question. The Apple ban is not directly addressed.

Step 4 - Evasion check: There is partial evasion and deflection. Instead of directly evaluating Xi's sincerity, the speaker pivots to their own sincerity and general comments on Xi's economic difficulties.

Summary: The respondent partially addresses the topic but does not clearly answer whether Xi is sincere. The answer shifts focus to the speaker's own sincerity rather than giving a straightforward judgment about Xi's sincerity.

LABEL: Ambivalent"""
    },
    # Example 3: Clear Non-Reply
    {
        "role": "user",
        "content": """Based on a segment of the interview where the interviewer asks a series of questions, classify the type of response provided by the interviewee for the following question.

### Classification Categories ###
1. Clear Reply - The information requested is explicitly stated (in the requested form)
2. Clear Non-Reply - The information requested is not given at all due to ignorance, need for clarification, or declining to answer
3. Ambivalent - The information requested is given in an incomplete way (e.g., the answer is too general, partial, implicit, dodging, or deflection)

### Full Interview Question ###
Q. What did you talk about with him?

### Full Interview Answer ###
We talked about what we talked about at the conference overall. We talked about stability. We talked about making sure that the Southern Hemisphere had access to change. Thank you, everybody. This ends the press conference.

### Specific Question to Classify ###
What did you talk about with him?

Think step by step, then provide your classification."""
    },
    {
        "role": "assistant",
        "content": """Step 1 - What the question asks: The question is asking for the specific substantive content of a conversation with a particular person ("him"): concrete topics, issues, or points that were discussed.

Step 2 - How the answer addresses it: The answer does not provide clear, specific content of what was discussed. It uses vague, circular phrases ("what we talked about at the conference overall," "stability") without specifying what was actually said, then devolves into closing remarks.

Step 3 - Information type: The requested information is effectively missing. While there are some topic-like words ("stability"), they are not tied to a concrete description of the conversation.

Step 4 - Evasion check: Yes, the respondent effectively evades the question. They start with a circular phrase ("We talked about what we talked about at the conference overall"), then the response breaks into closing remarks.

Summary: The question seeks a clear description of the content of a specific conversation. The answer offers only vague, generic themes and never concretely states what was discussed. The requested information is not provided at all.

LABEL: Clear Non-Reply"""
    },
]


def load_dataset_from_huggingface() -> pd.DataFrame:
    """Load the QEvasion test dataset from HuggingFace."""
    print(f"Loading dataset from HuggingFace: {HF_DATASET}")
    df = pd.read_parquet(f"hf://datasets/{HF_DATASET}/{HF_TEST_SPLIT}")
    print(f"Loaded {len(df)} examples")
    return df


def load_dataset_from_csv(path: str) -> pd.DataFrame:
    """Load dataset from local CSV file."""
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} examples")
    return df


def extract_label(response: str) -> str:
    """Extract the final label from model response."""
    # Try to find LABEL pattern
    match = re.search(r'LABEL:\s*(Clear Reply|Clear Non-Reply|Ambivalent)', response, re.IGNORECASE)
    if match:
        label_map = {
            "clear reply": "Clear Reply",
            "clear non-reply": "Clear Non-Reply",
            "ambivalent": "Ambivalent"
        }
        return label_map.get(match.group(1).lower(), match.group(1))
    
    # Fallback: check last 200 chars for mentions
    last_part = response[-200:].lower()
    if "clear reply" in last_part and "non-reply" not in last_part:
        return "Clear Reply"
    elif "non-reply" in last_part:
        return "Clear Non-Reply"
    elif "ambivalent" in last_part:
        return "Ambivalent"
    
    return "PARSE_ERROR"


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate text to maximum characters."""
    if pd.isna(text):
        return ""
    text = str(text)
    if len(text) > max_chars:
        return text[:max_chars] + "... [truncated]"
    return text


def predict_single(
    client: OpenAI,
    row: pd.Series,
    model: str,
    temperature: float = 0.1,
    use_reasoning: bool = False,
    max_retries: int = 3,
) -> dict:
    """Run prediction on a single example using OpenRouter."""
    
    # Build user prompt for the current example
    user_content = USER_TEMPLATE.format(
        interview_question=truncate_text(row['interview_question'], 1500),
        interview_answer=truncate_text(row['interview_answer'], 2500),
        question=truncate_text(row['question'], 500),
    )
    
    # Build messages with few-shot examples
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add few-shot examples as conversation turns
    messages.extend(FEW_SHOT_EXAMPLES)
    
    # Add the current example to classify
    messages.append({"role": "user", "content": user_content})
    
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
            
            message = response.choices[0].message
            content = message.content or ""
            
            # Extract reasoning details if available
            reasoning = ""
            if use_reasoning and hasattr(message, 'reasoning_details') and message.reasoning_details:
                reasoning = str(message.reasoning_details)
            
            # Extract label from response
            prediction = extract_label(content)
            
            return {
                "prediction": prediction,
                "full_response": content,
                "reasoning": reasoning,
            }
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # Exponential backoff
                print(f"\n    API error (attempt {attempt + 1}/{max_retries}):")
                print(f"      Type: {error_type}")
                print(f"      Message: {error_msg}")
                
                # Print additional details for specific error types
                if hasattr(e, 'response'):
                    print(f"      Response status: {getattr(e.response, 'status_code', 'N/A')}")
                    try:
                        print(f"      Response body: {e.response.text[:500]}")
                    except Exception:
                        pass
                if hasattr(e, 'request'):
                    print(f"      Request URL: {getattr(e.request, 'url', 'N/A')}")
                
                print(f"      Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n    FINAL ERROR after {max_retries} attempts:")
                print(f"      Type: {error_type}")
                print(f"      Message: {error_msg}")
                print(f"      Traceback:\n{traceback.format_exc()}")
                
                return {
                    "prediction": "ERROR",
                    "full_response": f"{error_type}: {error_msg}\n{traceback.format_exc()}",
                    "reasoning": "",
                }


def run_predictions(
    client: OpenAI,
    df: pd.DataFrame,
    model: str,
    temperature: float = 0.1,
    use_reasoning: bool = False,
    rate_limit_delay: float = 1.0,
    checkpoint_every: int = 50,
    output_path: str = "predictions.csv",
) -> list:
    """Run predictions on the entire dataset."""
    
    print("\n" + "=" * 60)
    print(f"Running predictions with model: {model}")
    print(f"Temperature: {temperature}, Reasoning: {use_reasoning}")
    print(f"Rate limit delay: {rate_limit_delay}s")
    print("=" * 60 + "\n")
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        result = predict_single(
            client,
            row,
            model=model,
            temperature=temperature,
            use_reasoning=use_reasoning,
        )
        
        results.append({
            "index": idx,
            "question": row['question'],
            "interview_question": row['interview_question'],
            "interview_answer": row['interview_answer'],
            "prediction": result['prediction'],
            "raw_response": result['full_response'],
            "reasoning_details": result['reasoning'],
            "clarity_label": row.get('clarity_label', ''),
        })
        
        # Checkpoint: save intermediate results
        if (idx + 1) % checkpoint_every == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(output_path.replace('.csv', '_checkpoint.csv'), index=False)
            print(f"\n  Checkpoint saved at {idx + 1} examples")
        
        # Rate limiting
        time.sleep(rate_limit_delay)
    
    return results


def print_summary(results: list, output_file: str):
    """Print summary statistics and evaluation metrics."""
    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60 + "\n")
    
    total = len(results)
    
    # Count categories
    parse_errors = sum(1 for r in results if r['prediction'] == 'PARSE_ERROR')
    errors = sum(1 for r in results if r['prediction'] == 'ERROR')
    successful = total - parse_errors - errors
    
    print(f"Total examples: {total}")
    print(f"  Successful:    {successful} ({successful/total*100:.1f}%)")
    print(f"  Parse errors:  {parse_errors} ({parse_errors/total*100:.1f}%)")
    print(f"  API errors:    {errors} ({errors/total*100:.1f}%)")
    
    # Prediction distribution
    print("\nPrediction distribution:")
    for label in ["Clear Reply", "Clear Non-Reply", "Ambivalent"]:
        count = sum(1 for r in results if r['prediction'] == label)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Evaluation metrics if ground truth available
    has_labels = any(r.get('clarity_label') for r in results)
    if has_labels:
        # Filter to valid predictions with ground truth
        valid_results = [
            r for r in results 
            if r['prediction'] not in ['PARSE_ERROR', 'ERROR'] 
            and r.get('clarity_label')
        ]
        
        if valid_results:
            y_true = [r['clarity_label'] for r in valid_results]
            y_pred = [r['prediction'] for r in valid_results]
            
            labels = ["Clear Reply", "Clear Non-Reply", "Ambivalent"]
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision_macro = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
            recall_macro = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
            
            print("\n" + "=" * 60)
            print("EVALUATION METRICS")
            print("=" * 60)
            print(f"\nValid predictions: {len(valid_results)}/{total}")
            print("\nOverall Metrics:")
            print(f"  Accuracy:        {accuracy*100:.2f}%")
            print(f"  Precision (macro): {precision_macro*100:.2f}%")
            print(f"  Recall (macro):    {recall_macro*100:.2f}%")
            print(f"  F1 Score (macro):  {f1_macro*100:.2f}%")
            
            # Per-class metrics
            print("\nPer-class Classification Report:")
            print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict on HuggingFace QEvasion dataset using OpenRouter"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="arcee-ai/trinity-large-preview:free",
        help="OpenRouter model to use (default: arcee-ai/trinity-large-preview:free)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to local CSV file (if not provided, loads from HuggingFace)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="openrouter_predictions.csv",
        help="Output CSV path"
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
        help="Enable reasoning mode (OpenRouter extended thinking)"
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
        default=50,
        help="Save checkpoint every N examples (default: 50)"
    )
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    # Initialize OpenRouter client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    # Load dataset
    if args.data:
        df = load_dataset_from_csv(args.data)
    else:
        df = load_dataset_from_huggingface()
    
    if args.limit:
        print(f"Limiting to first {args.limit} examples")
        df = df.head(args.limit)
    
    # Run predictions
    results = run_predictions(
        client=client,
        df=df,
        model=args.model,
        temperature=args.temperature,
        use_reasoning=args.reasoning,
        rate_limit_delay=args.rate_limit,
        checkpoint_every=args.checkpoint_every,
        output_path=args.output,
    )
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    
    # Clean up checkpoint file if it exists
    checkpoint_path = args.output.replace('.csv', '_checkpoint.csv')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed checkpoint file: {checkpoint_path}")
    
    # Print summary
    print_summary(results, args.output)


if __name__ == "__main__":
    main()
