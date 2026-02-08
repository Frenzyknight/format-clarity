"""
GPT-5.2 Two-Stage Judge System for Clarity Classification

This script implements a two-stage evaluation system using the GPT-5.2 Responses API:
1. First pass: Query GPT-5.2 with default reasoning (none) for initial predictions
2. Compare with reference predictions file
3. Second pass: For disagreements, re-query GPT-5.2 with xhigh reasoning + few-shot approach

Uses the Responses API for improved intelligence through chain-of-thought passing.
"""

import argparse
import os
import re
import time
from typing import Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# GPT-5.2 CONFIGURATION
# ============================================================================

# Model variants available:
# - gpt-5.2: Complex reasoning, broad world knowledge (default)
# - gpt-5.2-pro: Tough problems requiring harder thinking
# - gpt-5-mini: Cost-optimized, balances speed/cost/capability
# - gpt-5-nano: High-throughput simple tasks

DEFAULT_MODEL = "gpt-5.2"
HIGH_REASONING_MODEL = "gpt-5.2-pro"  # Use gpt-5.2-pro for harder thinking on disagreements

# Reasoning effort levels for GPT-5.2:
# none (default), low, medium, high, xhigh
FIRST_PASS_REASONING = "low"  # Lowest latency for fast initial classification
SECOND_PASS_REASONING = "high"  # Maximum reasoning for disagreements

# Verbosity levels: low, medium (default), high
# For classification tasks, low verbosity is sufficient - we just need the label
# Reasoning effort (xhigh) handles the deep thinking, verbosity just controls output length
FIRST_PASS_VERBOSITY = "low"  # Concise output for classification
SECOND_PASS_VERBOSITY = "low"  # Still low - reasoning effort handles the thinking


# ============================================================================
# PROMPTS - First Pass (Standard)
# ============================================================================

FIRST_PASS_SYSTEM_PROMPT = """You are an expert political discourse analyst. Your task is to classify political interview responses into one of three categories.

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

### Output Format ###
Provide your response in the following format:
REASONING: [Brief analysis]
LABEL: [Clear Reply|Clear Non-Reply|Ambivalent]"""

FIRST_PASS_USER_TEMPLATE = """Classify the following political interview response:

### Interview Question ###
{interview_question}

### Interview Answer ###
{interview_answer}

### Specific Question Being Classified ###
{question}

Analyze the response and provide your classification."""


# ============================================================================
# PROMPTS - Second Pass (High Reasoning + Few-Shot)
# ============================================================================

HIGH_REASONING_SYSTEM_PROMPT = """You are a senior expert political discourse analyst with deep expertise in identifying evasion, ambiguity, and clarity in political communication. Your task requires careful, thorough analysis.

### Classification Categories (Detailed) ###

1. **Clear Reply** - The information requested is EXPLICITLY and COMPLETELY stated:
   - The answer must directly address the specific question asked
   - Specific, concrete information must be provided
   - The response should leave no ambiguity about the answer
   - Example: Q: "Do you support the bill?" A: "Yes, I fully support this bill."

2. **Clear Non-Reply** - The information requested is NOT given at all:
   - Explicit admission of ignorance: "I don't know", "I'm not sure"
   - Request for clarification: "What do you mean by...?", "Could you be more specific?"
   - Explicit refusal: "I can't comment", "That's classified", "I decline to answer"
   - The key is that NO attempt is made to answer the question
   - Example: Q: "What is the cost?" A: "I don't have those numbers with me."

3. **Ambivalent** - The information is given INCOMPLETELY or INDIRECTLY:
   - Too general: Broad statements that technically address the topic but lack specifics
   - Partial: Only part of the question is addressed
   - Implicit: The answer can be inferred but is not stated directly
   - Dodging: Pivoting to a related but different topic
   - Deflection: Answering a different question than the one asked
   - Conditional answers: "It depends on...", "Under certain circumstances..."
   - Example: Q: "Do you support the bill?" A: "I believe we need to carefully consider all aspects of this legislation."

### Critical Analysis Guidelines ###
- Focus on whether the SPECIFIC question asked is answered, not the general topic
- Politicians often appear to answer while actually evading - watch for this
- Consider what information was requested vs what was actually provided
- Be precise in distinguishing between "no answer attempted" (Clear Non-Reply) and "partial/evasive answer" (Ambivalent)

### Output Format ###
DETAILED_ANALYSIS: [Thorough step-by-step analysis of:
1. What specific information is the question requesting?
2. What information, if any, is provided in the answer?
3. Does the answer directly, partially, or not at all address the specific request?]

FINAL_LABEL: [Clear Reply|Clear Non-Reply|Ambivalent]"""


FEW_SHOT_EXAMPLES = """
### Example 1 ###
**Interview Question**: Mr. President, have you any ideas on what should be done with the evidence turned up on Senator McCarthy by that subcommittee?
**Interview Answer**: I understand that the Justice Department is making an investigation. It has been referred to the Justice Department, so I have no comment to make on it.
**Question to Classify**: Mr. President, have you any ideas on what should be done with the evidence turned up on Senator McCarthy by that subcommittee?

**Analysis**: The question asks for the President's IDEAS about what should be done. The answer states that the matter has been referred to the Justice Department and declines to comment. This is an explicit refusal to provide the requested information (his ideas) - he is declining to answer.
**Correct Label**: Clear Non-Reply

---

### Example 2 ###
**Interview Question**: Do you think the advantages all around would outweigh the risks or the embarrassments?
**Interview Answer**: Oh, yes, I think so. I have had all sorts of well, in 324 press conferences I imagine I have had all the experiences that a man can possibly have at a press conference, and I have never felt that I would want to discontinue them. And I have never felt that I have been unfairly treated.
**Question to Classify**: Do you think the advantages all around would outweigh the risks or the embarrassments?

**Analysis**: The question asks whether advantages outweigh risks. The answer directly states "Oh, yes, I think so" - providing a clear, explicit affirmative answer to the specific question, followed by supporting explanation.
**Correct Label**: Clear Reply

---

### Example 3 ###
**Interview Question**: Does the Prime Minister feel that that would have a major impact on the peace talks?
**Interview Answer**: The President: Nice try. The Prime Minister: I don't deal with domestic political American or personal domestic American problems.
**Question to Classify**: Does the Prime Minister feel that that would have a major impact on the peace talks?

**Analysis**: The question asks about the Prime Minister's opinion on impact on peace talks. The Prime Minister's response deflects by claiming it's a domestic American issue and refuses to engage with the substance of the question. However, this is not simply "I don't know" or "I can't comment" - it's an active avoidance through reframing. This is deflection/dodging.
**Correct Label**: Ambivalent

---

### Example 4 ###
**Interview Question**: Are you going to do it now?
**Interview Answer**: I can not tell you that at the moment. We'll have to see how things develop.
**Question to Classify**: Are you going to do it now?

**Analysis**: The question asks about a specific action timing. The answer says "I cannot tell you" - but this is not the same as "I don't know." It's a refusal to commit combined with deferral ("We'll have to see"). This is evasive/partial - acknowledging the question but not providing a clear yes/no answer.
**Correct Label**: Ambivalent

---

### Example 5 ###
**Interview Question**: When is that?
**Interview Answer**: I cannot say at this time. The schedule has not been finalized.
**Question to Classify**: When is that?

**Analysis**: The question asks for a specific time. The answer explicitly states inability to provide the information because the schedule isn't finalized. This is a clear admission that they don't have/can't provide the requested information - not evasion, but genuine inability to answer.
**Correct Label**: Clear Non-Reply

"""


HIGH_REASONING_USER_TEMPLATE = """You are reviewing a case where two classification systems disagreed. Apply deep analysis to determine the correct classification.

{few_shot_examples}

---

### Current Case to Classify ###

**Interview Question**: {interview_question}

**Interview Answer**: {interview_answer}

**Question to Classify**: {question}

**Previous Classifications**:
- Reference System: {reference_prediction}
- First Pass: {first_pass_prediction}

NOTE: The previous systems disagreed. You must make your OWN independent judgment based on careful analysis of the content above and the classification guidelines. Do not simply pick one of the previous answers - analyze from scratch.

Provide your detailed analysis and final classification."""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate text to maximum characters."""
    if pd.isna(text):
        return ""
    text = str(text)
    if len(text) > max_chars:
        return text[:max_chars] + "... [truncated]"
    return text


def extract_label(response: str, strict: bool = False) -> str:
    """Extract the final label from a response."""
    # Try to find FINAL_LABEL pattern (used in high reasoning)
    match = re.search(r'FINAL_LABEL:\s*(Clear Reply|Clear Non-Reply|Ambivalent)', response, re.IGNORECASE)
    if match:
        return normalize_label_case(match.group(1))
    
    # Try to find LABEL pattern (used in first pass)
    match = re.search(r'LABEL:\s*(Clear Reply|Clear Non-Reply|Ambivalent)', response, re.IGNORECASE)
    if match:
        return normalize_label_case(match.group(1))
    
    if strict:
        return "PARSE_ERROR"
    
    # Fallback: check last part of response
    last_part = response[-300:].lower()
    if "clear reply" in last_part and "non-reply" not in last_part:
        return "Clear Reply"
    elif "non-reply" in last_part:
        return "Clear Non-Reply"
    elif "ambivalent" in last_part:
        return "Ambivalent"
    
    return "PARSE_ERROR"


def normalize_label_case(label: str) -> str:
    """Normalize label to proper case."""
    label_map = {
        "clear reply": "Clear Reply",
        "clear non-reply": "Clear Non-Reply",
        "ambivalent": "Ambivalent"
    }
    return label_map.get(label.lower().strip(), label)


def normalize_label(label: str) -> str:
    """Normalize label for comparison."""
    if pd.isna(label) or label == '':
        return 'MISSING'
    label = str(label).strip().lower()
    if 'clear reply' in label and 'non' not in label:
        return 'Clear Reply'
    elif 'non-reply' in label or 'non reply' in label:
        return 'Clear Non-Reply'
    elif 'ambivalent' in label:
        return 'Ambivalent'
    return 'UNKNOWN'


# ============================================================================
# API FUNCTIONS - GPT-5.2 Responses API
# ============================================================================

def query_gpt52(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = FIRST_PASS_REASONING,
    verbosity: str = FIRST_PASS_VERBOSITY,
    max_retries: int = 3,
    previous_response_id: Optional[str] = None,
) -> dict:
    """
    Query GPT-5.2 using the Responses API.
    
    Args:
        client: OpenAI client
        system_prompt: System instructions
        user_prompt: User message
        model: Model to use (gpt-5.2, gpt-5.2-pro, gpt-5-mini, etc.)
        reasoning_effort: none, low, medium, high, xhigh
        verbosity: low, medium, high
        max_retries: Number of retry attempts
        previous_response_id: ID of previous response for CoT passing
    
    Returns:
        dict with success status, response text, and metadata
    """
    
    for attempt in range(max_retries):
        try:
            # Build the input with system and user prompts
            input_content = f"{system_prompt}\n\n---\n\n{user_prompt}"
            
            # Build request parameters for Responses API
            request_params = {
                "model": model,
                "input": input_content,
                "reasoning": {
                    "effort": reasoning_effort
                },
                "text": {
                    "verbosity": verbosity
                }
            }
            
            # Pass previous response ID for CoT continuity if available
            if previous_response_id:
                request_params["previous_response_id"] = previous_response_id
            
            # Use the Responses API
            response = client.responses.create(**request_params)
            
            # Extract the output text from the response
            response_text = response.output_text if hasattr(response, 'output_text') else str(response.output)
            
            return {
                "success": True,
                "response": response_text,
                "model": model,
                "response_id": response.id if hasattr(response, 'id') else None,
                "reasoning_effort": reasoning_effort,
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"    API error (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return {
                    "success": False,
                    "response": str(e),
                    "model": model,
                    "response_id": None,
                    "reasoning_effort": reasoning_effort,
                }


def query_gpt52_chat_fallback(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = FIRST_PASS_REASONING,
    verbosity: str = FIRST_PASS_VERBOSITY,
    max_retries: int = 3,
) -> dict:
    """
    Fallback to Chat Completions API if Responses API is unavailable.
    
    Uses the Chat Completions format with GPT-5.2 specific parameters.
    """
    
    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Chat Completions API format for GPT-5.2
            request_params = {
                "model": model,
                "messages": messages,
                "reasoning_effort": reasoning_effort,
                "verbosity": verbosity,
            }
            
            response = client.chat.completions.create(**request_params)
            response_text = response.choices[0].message.content
            
            return {
                "success": True,
                "response": response_text,
                "model": model,
                "response_id": None,
                "reasoning_effort": reasoning_effort,
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"    API error (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return {
                    "success": False,
                    "response": str(e),
                    "model": model,
                    "response_id": None,
                    "reasoning_effort": reasoning_effort,
                }


def query_gpt52_auto(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = FIRST_PASS_REASONING,
    verbosity: str = FIRST_PASS_VERBOSITY,
    max_retries: int = 3,
    previous_response_id: Optional[str] = None,
    use_responses_api: bool = True,
) -> dict:
    """
    Auto-select between Responses API and Chat Completions fallback.
    
    Tries Responses API first, falls back to Chat Completions if needed.
    """
    if use_responses_api:
        result = query_gpt52(
            client, system_prompt, user_prompt, model,
            reasoning_effort, verbosity, max_retries, previous_response_id
        )
        if result["success"]:
            return result
        # If Responses API fails, try Chat Completions fallback
        print("    Responses API failed, trying Chat Completions fallback...")
    
    return query_gpt52_chat_fallback(
        client, system_prompt, user_prompt, model,
        reasoning_effort, verbosity, max_retries
    )


# ============================================================================
# FIRST PASS - Initial Classification (reasoning: none, verbosity: low)
# ============================================================================

def run_first_pass(
    client: OpenAI,
    eval_df: pd.DataFrame,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = FIRST_PASS_REASONING,
    verbosity: str = FIRST_PASS_VERBOSITY,
    rate_limit_delay: float = 0.3,
    use_responses_api: bool = True,
) -> pd.DataFrame:
    """
    Run first pass classification on all examples.
    
    Uses GPT-5.2 with minimal reasoning for fast initial classification.
    Reasoning effort 'none' provides lower-latency interactions.
    """
    
    print("\n" + "=" * 70)
    print("FIRST PASS: Initial GPT-5.2 Classification")
    print(f"Model: {model}")
    print(f"Reasoning Effort: {reasoning_effort} | Verbosity: {verbosity}")
    print("=" * 70 + "\n")
    
    results = []
    
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="First Pass"):
        user_prompt = FIRST_PASS_USER_TEMPLATE.format(
            interview_question=truncate_text(row['interview_question'], 1000),
            interview_answer=truncate_text(row['interview_answer'], 2000),
            question=truncate_text(row['question'], 500),
        )
        
        result = query_gpt52_auto(
            client,
            FIRST_PASS_SYSTEM_PROMPT,
            user_prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            use_responses_api=use_responses_api,
        )
        
        if result['success']:
            label = extract_label(result['response'])
        else:
            label = "ERROR"
        
        results.append({
            'index': row.get('index', idx),
            'gpt_first_prediction': label,
            'gpt_first_response': result['response'],
            'response_id': result.get('response_id'),
        })
        
        time.sleep(rate_limit_delay)
    
    results_df = pd.DataFrame(results)
    return results_df


# ============================================================================
# COMPARISON AND DISAGREEMENT DETECTION
# ============================================================================

def find_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    """Find rows where GPT first pass and reference predictions disagree."""
    
    # Normalize for comparison
    df['gpt_norm'] = df['gpt_first_prediction'].apply(normalize_label)
    df['ref_norm'] = df['reference_prediction'].apply(normalize_label)
    
    # Find disagreements (excluding errors and missing values)
    valid_mask = (
        (df['gpt_norm'].isin(['Clear Reply', 'Clear Non-Reply', 'Ambivalent'])) &
        (df['ref_norm'].isin(['Clear Reply', 'Clear Non-Reply', 'Ambivalent']))
    )
    
    disagreements = df[valid_mask & (df['gpt_norm'] != df['ref_norm'])].copy()
    
    return disagreements


# ============================================================================
# SECOND PASS - xHigh Reasoning + Few-Shot for Disagreements
# ============================================================================

def run_second_pass(
    client: OpenAI,
    df: pd.DataFrame,
    disagreements: pd.DataFrame,
    model: str = HIGH_REASONING_MODEL,
    reasoning_effort: str = SECOND_PASS_REASONING,
    verbosity: str = SECOND_PASS_VERBOSITY,
    rate_limit_delay: float = 0.5,
    include_few_shot: bool = True,
    use_responses_api: bool = True,
) -> pd.DataFrame:
    """
    Run second pass with xhigh reasoning on disagreements.
    
    Uses GPT-5.2 with maximum reasoning effort (xhigh) for thorough analysis.
    The xhigh setting is new in GPT-5.2 and provides the deepest reasoning.
    
    High verbosity ensures thorough explanations for complex classification cases.
    """
    
    print("\n" + "=" * 70)
    print("SECOND PASS: xHigh Reasoning + Few-Shot on Disagreements")
    print(f"Model: {model}")
    print(f"Reasoning Effort: {reasoning_effort} | Verbosity: {verbosity}")
    print(f"Disagreements to analyze: {len(disagreements)}")
    print("=" * 70 + "\n")
    
    # Initialize columns
    df['gpt_second_prediction'] = 'N/A'
    df['gpt_second_response'] = ''
    df['final_prediction'] = df['gpt_first_prediction']
    
    if len(disagreements) == 0:
        print("No disagreements to process!")
        return df
    
    few_shot = FEW_SHOT_EXAMPLES if include_few_shot else ""
    
    for idx, row in tqdm(disagreements.iterrows(), total=len(disagreements), desc="Second Pass (xhigh)"):
        user_prompt = HIGH_REASONING_USER_TEMPLATE.format(
            few_shot_examples=few_shot,
            interview_question=truncate_text(row['interview_question'], 1500),
            interview_answer=truncate_text(row['interview_answer'], 3000),
            question=truncate_text(row['question'], 500),
            reference_prediction=row['reference_prediction'],
            first_pass_prediction=row['gpt_first_prediction'],
        )
        
        result = query_gpt52_auto(
            client,
            HIGH_REASONING_SYSTEM_PROMPT,
            user_prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            use_responses_api=use_responses_api,
        )
        
        if result['success']:
            label = extract_label(result['response'])
        else:
            label = "ERROR"
        
        df.loc[idx, 'gpt_second_prediction'] = label
        df.loc[idx, 'gpt_second_response'] = result['response']
        
        # Set final prediction from second pass (xhigh reasoning result)
        if label not in ['ERROR', 'PARSE_ERROR']:
            df.loc[idx, 'final_prediction'] = label
        
        time.sleep(rate_limit_delay)
    
    return df


# ============================================================================
# EVALUATION AND REPORTING
# ============================================================================

def print_summary(df: pd.DataFrame, disagreements: pd.DataFrame):
    """Print comprehensive summary statistics."""
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    total = len(df)
    num_disagreements = len(disagreements)
    
    print(f"\nTotal examples: {total}")
    print(f"Initial agreements: {total - num_disagreements} ({(total - num_disagreements)/total*100:.1f}%)")
    print(f"Disagreements requiring second pass: {num_disagreements} ({num_disagreements/total*100:.1f}%)")
    
    # Check if we have ground truth labels
    has_labels = 'clarity_label' in df.columns
    
    if has_labels:
        print("\n" + "-" * 50)
        print("ACCURACY METRICS (vs Ground Truth)")
        print("-" * 50)
        
        # Normalize all labels first
        df['gt_norm'] = df['clarity_label'].apply(normalize_label)
        df['final_norm'] = df['final_prediction'].apply(normalize_label)
        
        # Filter to valid ground truth
        valid_gt = df[df['gt_norm'].isin(['Clear Reply', 'Clear Non-Reply', 'Ambivalent'])]
        
        if len(valid_gt) > 0:
            # Reference accuracy
            ref_correct = sum(valid_gt['ref_norm'] == valid_gt['gt_norm'])
            ref_acc = ref_correct / len(valid_gt)
            print(f"\nReference System Accuracy: {ref_acc:.4f} ({ref_acc*100:.2f}%)")
            
            # First pass accuracy
            gpt_first_correct = sum(valid_gt['gpt_norm'] == valid_gt['gt_norm'])
            gpt_first_acc = gpt_first_correct / len(valid_gt)
            print(f"GPT First Pass Accuracy: {gpt_first_acc:.4f} ({gpt_first_acc*100:.2f}%)")
            
            # Final prediction accuracy
            final_correct = sum(valid_gt['final_norm'] == valid_gt['gt_norm'])
            final_acc = final_correct / len(valid_gt)
            print(f"Final (After Second Pass) Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
            
            # Improvement
            improvement = final_acc - max(ref_acc, gpt_first_acc)
            print(f"\nImprovement over best single system: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    # Second pass statistics
    if num_disagreements > 0:
        print("\n" + "-" * 50)
        print("SECOND PASS DECISIONS")
        print("-" * 50)
        
        judged = df[df['gpt_second_prediction'] != 'N/A']
        
        agreed_with_ref = sum(judged['gpt_second_prediction'] == judged['reference_prediction'])
        agreed_with_first = sum(judged['gpt_second_prediction'] == judged['gpt_first_prediction'])
        overruled_both = num_disagreements - agreed_with_ref - agreed_with_first
        
        print(f"\nOn {num_disagreements} disagreements:")
        print(f"  Agreed with Reference: {agreed_with_ref} ({agreed_with_ref/num_disagreements*100:.1f}%)")
        print(f"  Agreed with First Pass: {agreed_with_first} ({agreed_with_first/num_disagreements*100:.1f}%)")
        print(f"  Different from both: {overruled_both} ({overruled_both/num_disagreements*100:.1f}%)")
    
    # Final prediction distribution
    print("\n" + "-" * 50)
    print("FINAL PREDICTION DISTRIBUTION")
    print("-" * 50)
    
    for label in ["Clear Reply", "Clear Non-Reply", "Ambivalent"]:
        count = sum(df['final_prediction'].apply(normalize_label) == label)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPT-5.2 Two-Stage Judge System for Clarity Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with GPT-5.2
  python gpt52_judge.py

  # Use GPT-5.2-pro for harder thinking on disagreements
  python gpt52_judge.py --high-reasoning-model gpt-5.2-pro

  # Test with limited examples
  python gpt52_judge.py --limit 50

  # Custom first pass reasoning level
  python gpt52_judge.py --first-reasoning medium

  # Use Chat Completions API fallback
  python gpt52_judge.py --use-chat-api

Reasoning Effort Levels (GPT-5.2):
  none   - Default, lowest latency (first pass default)
  low    - Minimal reasoning
  medium - Moderate reasoning
  high   - Thorough reasoning
  xhigh  - Maximum reasoning (second pass default)
        """
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="clarity_task_evaluation_dataset.csv",
        help="Path to evaluation dataset CSV"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="llama.csv",
        help="Path to reference predictions CSV (with 'prediction' column)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gpt52_judge_results.csv",
        help="Final output CSV path"
    )
    parser.add_argument(
        "--first-pass-output",
        type=str,
        default="gpt52_first_pass.csv",
        help="Output CSV for first pass results (saved before second pass)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"GPT-5.2 model variant (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--high-reasoning-model",
        type=str,
        default=HIGH_REASONING_MODEL,
        help=f"Model for second pass (default: {HIGH_REASONING_MODEL})"
    )
    parser.add_argument(
        "--first-reasoning",
        type=str,
        default=FIRST_PASS_REASONING,
        choices=["none", "low", "medium", "high", "xhigh"],
        help=f"Reasoning effort for first pass (default: {FIRST_PASS_REASONING})"
    )
    parser.add_argument(
        "--second-reasoning",
        type=str,
        default=SECOND_PASS_REASONING,
        choices=["none", "low", "medium", "high", "xhigh"],
        help=f"Reasoning effort for second pass (default: {SECOND_PASS_REASONING})"
    )
    parser.add_argument(
        "--first-verbosity",
        type=str,
        default=FIRST_PASS_VERBOSITY,
        choices=["low", "medium", "high"],
        help=f"Output verbosity for first pass (default: {FIRST_PASS_VERBOSITY}). Low is fine for classification."
    )
    parser.add_argument(
        "--second-verbosity",
        type=str,
        default=SECOND_PASS_VERBOSITY,
        choices=["low", "medium", "high"],
        help=f"Output verbosity for second pass (default: {SECOND_PASS_VERBOSITY}). Reasoning effort handles thinking."
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.3,
        help="Delay between API calls in seconds (default: 0.3)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)"
    )
    parser.add_argument(
        "--skip-first-pass",
        action="store_true",
        help="Skip first pass and load from existing results"
    )
    parser.add_argument(
        "--first-pass-results",
        type=str,
        default=None,
        help="Path to existing first pass results (used with --skip-first-pass)"
    )
    parser.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Disable few-shot examples in second pass"
    )
    parser.add_argument(
        "--use-chat-api",
        action="store_true",
        help="Use Chat Completions API instead of Responses API"
    )
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Load evaluation data
    print("\n" + "=" * 70)
    print("GPT-5.2 Two-Stage Judge System")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  First Pass:  model={args.model}, reasoning={args.first_reasoning}, verbosity={args.first_verbosity}")
    high_model = args.high_reasoning_model or args.model
    print(f"  Second Pass: model={high_model}, reasoning={args.second_reasoning}, verbosity={args.second_verbosity}")
    print(f"  API: {'Chat Completions' if args.use_chat_api else 'Responses API'}")
    
    print("\n" + "-" * 70)
    print("Loading Data")
    print("-" * 70)
    
    print(f"\n  Evaluation data: {args.eval_data}")
    eval_df = pd.read_csv(args.eval_data)
    print(f"    Loaded {len(eval_df)} examples")
    
    # Load reference predictions
    print(f"\n  Reference predictions: {args.reference}")
    ref_df = pd.read_csv(args.reference)
    print(f"    Loaded {len(ref_df)} predictions")
    
    # Align data
    min_len = min(len(eval_df), len(ref_df))
    print(f"\n  Using {min_len} examples (minimum of both)")
    
    eval_df = eval_df.head(min_len).reset_index(drop=True)
    ref_df = ref_df.head(min_len).reset_index(drop=True)
    
    # Apply limit if specified
    if args.limit:
        print(f"  Limiting to first {args.limit} examples")
        eval_df = eval_df.head(args.limit)
        ref_df = ref_df.head(args.limit)
    
    # Add reference predictions to eval dataframe
    eval_df['reference_prediction'] = ref_df['prediction']
    if 'clarity_label' in ref_df.columns:
        eval_df['clarity_label'] = ref_df['clarity_label']
    
    # First Pass
    use_responses_api = not args.use_chat_api
    
    if args.skip_first_pass and args.first_pass_results:
        print(f"\n  Loading first pass results from: {args.first_pass_results}")
        first_pass_df = pd.read_csv(args.first_pass_results)
        eval_df['gpt_first_prediction'] = first_pass_df['gpt_first_prediction']
        eval_df['gpt_first_response'] = first_pass_df.get('gpt_first_response', '')
    else:
        first_pass_df = run_first_pass(
            client,
            eval_df,
            model=args.model,
            reasoning_effort=args.first_reasoning,
            verbosity=args.first_verbosity,
            rate_limit_delay=args.rate_limit,
            use_responses_api=use_responses_api,
        )
        eval_df['gpt_first_prediction'] = first_pass_df['gpt_first_prediction']
        eval_df['gpt_first_response'] = first_pass_df['gpt_first_response']
        
        # Save first pass results immediately
        first_pass_columns = [
            'index', 'question', 'reference_prediction', 'gpt_first_prediction',
        ]
        if 'clarity_label' in eval_df.columns:
            first_pass_columns.append('clarity_label')
        first_pass_columns = [c for c in first_pass_columns if c in eval_df.columns]
        
        eval_df[first_pass_columns].to_csv(args.first_pass_output, index=False)
        print(f"\n✅ First pass saved to: {args.first_pass_output}")
        
        # Also save detailed first pass with full responses
        first_pass_detailed = args.first_pass_output.replace('.csv', '_detailed.csv')
        eval_df[['index', 'question', 'reference_prediction', 'gpt_first_prediction', 'gpt_first_response']].to_csv(
            first_pass_detailed, index=False
        )
        print(f"✅ First pass detailed saved to: {first_pass_detailed}")
    
    # Find disagreements
    disagreements = find_disagreements(eval_df)
    print(f"\n  Found {len(disagreements)} disagreements between GPT-5.2 and reference")
    
    # Second Pass with xHigh Reasoning
    eval_df = run_second_pass(
        client,
        eval_df,
        disagreements,
        model=high_model,
        reasoning_effort=args.second_reasoning,
        verbosity=args.second_verbosity,
        rate_limit_delay=args.rate_limit * 2,  # More delay for xhigh reasoning
        include_few_shot=not args.no_few_shot,
        use_responses_api=use_responses_api,
    )
    
    # Print summary
    print_summary(eval_df, disagreements)
    
    # Save results
    output_columns = [
        'index', 'question',
        'reference_prediction', 'gpt_first_prediction', 'gpt_second_prediction',
        'final_prediction',
    ]
    if 'clarity_label' in eval_df.columns:
        output_columns.append('clarity_label')
    
    output_columns = [c for c in output_columns if c in eval_df.columns]
    
    # Save main results
    eval_df[output_columns].to_csv(args.output, index=False)
    print(f"\n✅ Results saved to: {args.output}")
    
    # Save detailed results with full responses
    detailed_output = args.output.replace('.csv', '_detailed.csv')
    eval_df.to_csv(detailed_output, index=False)
    print(f"✅ Detailed results saved to: {detailed_output}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
