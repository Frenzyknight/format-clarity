"""
Generate Chain-of-Thought reasoning using GPT-4o Batch API.
50% cost savings compared to synchronous API calls.

USAGE:
1. export OPENAI_API_KEY="your-key"
2. python generate_cot_batch.py --create    # Create and submit batch
3. python generate_cot_batch.py --status    # Check batch status
4. python generate_cot_batch.py --download  # Download results when complete

OUTPUT:
- cot_data/train_cot.jsonl  -> Use this for Unsloth fine-tuning
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
# ============================================
# CONFIGURATION
# ============================================
OUTPUT_DIR = "./cot_data"
BATCH_INPUT_FILE = f"{OUTPUT_DIR}/batch_input.jsonl"
BATCH_STATUS_FILE = f"{OUTPUT_DIR}/batch_status.json"

# Batch size limit - very conservative to stay well under 900k token limit
# Each request is ~2000 tokens (input + output), so 100 requests ≈ 200k tokens
BATCH_SIZE = 100

# ============================================
load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# JSON Schema for Structured Outputs (Batch API compatible)
COT_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "cot_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "step1_question_asks": {
                    "type": "string",
                    "description": "What specific information is the question asking for?"
                },
                "step2_answer_addresses": {
                    "type": "string",
                    "description": "Does the answer directly address this specific question?"
                },
                "step3_information_type": {
                    "type": "string",
                    "description": "Is the information explicit, implicit, or missing entirely?"
                },
                "step4_evasion_check": {
                    "type": "string",
                    "description": "Does the respondent dodge, deflect, or give a partial answer?"
                },
                "reasoning_summary": {
                    "type": "string",
                    "description": "Overall reasoning summary explaining why the given classification is correct"
                }
            },
            "required": [
                "step1_question_asks",
                "step2_answer_addresses", 
                "step3_information_type",
                "step4_evasion_check",
                "reasoning_summary"
            ],
            "additionalProperties": False
        }
    }
}


def create_cot_prompt(interview_question: str, interview_answer: str, sub_question: str, ground_truth_label: str) -> str:
    """Create prompt for COT analysis based on ground truth label"""
    return f"""Analyze the following political interview segment and explain why the response is classified as "{ground_truth_label}" for the specific question.

### Classification Categories ###
1. Clear Reply - The information requested is explicitly stated (in the requested form)
2. Clear Non-Reply - The information requested is not given at all due to ignorance, need for clarification, or declining to answer  
3. Ambivalent - The information requested is given in an incomplete way (e.g., the answer is too general, partial, implicit, dodging, or deflection)

### Full Interview Question ###
{interview_question.strip()}

### Full Interview Answer ###
{interview_answer.strip()}

### Specific Question to Classify ###
{sub_question.strip()}

### Correct Classification ###
{ground_truth_label}

Analyze step by step and explain why this classification is correct."""


def create_batch_input():
    """Create batch input file from training data"""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Load training data
    splits = {'train': 'data/train-00000-of-00001.parquet'}
    print("Loading training data from HuggingFace: ailsntua/QEvasion")
    train_df = pd.read_parquet("hf://datasets/ailsntua/QEvasion/" + splits["train"])
    
    print(f"Training examples: {len(train_df)}")
    print(f"Label distribution:\n{train_df['clarity_label'].value_counts()}")
    
    # Save original data for later mapping
    train_df.to_parquet(f"{OUTPUT_DIR}/train_original.parquet")
    
    # Label normalization map
    label_map = {
        'clear reply': 'Clear Reply',
        'ambivalent': 'Ambivalent', 
        'clear non-reply': 'Clear Non-Reply',
        'clear non reply': 'Clear Non-Reply',
    }
    
    # Create batch requests
    batch_requests = []
    
    for idx, row in train_df.iterrows():
        # Normalize the ground truth label
        raw_label = str(row['clarity_label']).strip()
        ground_truth_label = label_map.get(raw_label.lower(), raw_label)
        
        prompt = create_cot_prompt(
            str(row['interview_question']),
            str(row['interview_answer']),
            str(row['question']),
            ground_truth_label
        )
        
        request = {
            "custom_id": f"request-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-5.1-2025-11-13",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert political discourse analyst specializing in classifying response clarity in political interviews. Analyze each question-answer pair step by step."
                    },
                    {"role": "user", "content": prompt}
                ],
                "response_format": COT_RESPONSE_SCHEMA,
                "temperature": 0.1,
            }
        }
        batch_requests.append(request)
    
    # Write batch input file
    with open(BATCH_INPUT_FILE, 'w') as f:
        for req in batch_requests:
            f.write(json.dumps(req) + '\n')
    
    print(f"Created batch input file: {BATCH_INPUT_FILE}")
    print(f"Total requests: {len(batch_requests)}")
    
    return BATCH_INPUT_FILE


def submit_batch(chunk_index: int | None = None):
    """Upload file and create batch job(s)
    
    Args:
        chunk_index: If provided, only submit this specific chunk (0-indexed).
                    If None, submit the next pending chunk.
    """
    
    # Create batch input if doesn't exist
    if not Path(BATCH_INPUT_FILE).exists():
        create_batch_input()
    
    # Load all requests
    with open(BATCH_INPUT_FILE, 'r') as f:
        all_requests = [json.loads(line) for line in f]
    
    total_requests = len(all_requests)
    total_chunks = (total_requests + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Total requests: {total_requests}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total chunks: {total_chunks}")
    
    # Load existing status or create new
    if Path(BATCH_STATUS_FILE).exists():
        with open(BATCH_STATUS_FILE, 'r') as f:
            status_info = json.load(f)
        # Migrate old format to new format
        if "batches" not in status_info:
            old_batch_id = status_info.get("batch_id")
            status_info = {
                "total_requests": total_requests,
                "total_chunks": total_chunks,
                "batch_size": BATCH_SIZE,
                "batches": {}
            }
            if old_batch_id:
                status_info["batches"]["0"] = {"batch_id": old_batch_id, "status": "unknown"}
    else:
        status_info = {
            "total_requests": total_requests,
            "total_chunks": total_chunks,
            "batch_size": BATCH_SIZE,
            "batches": {}
        }
    
    # Determine which chunk to submit
    if chunk_index is not None:
        chunks_to_submit = [chunk_index]
    else:
        # Find the next chunk that hasn't been submitted
        chunks_to_submit = []
        for i in range(total_chunks):
            if str(i) not in status_info["batches"]:
                chunks_to_submit = [i]
                break
        
        if not chunks_to_submit:
            print("\n✅ All chunks have been submitted!")
            print("Run 'python generate-cot.py --status' to check progress")
            return None
    
    for chunk_idx in chunks_to_submit:
        start_idx = chunk_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_requests)
        chunk_requests = all_requests[start_idx:end_idx]
        
        print(f"\n--- Submitting chunk {chunk_idx + 1}/{total_chunks} (requests {start_idx}-{end_idx - 1}) ---")
        
        # Write chunk to temp file
        chunk_file_path = f"{OUTPUT_DIR}/batch_input_chunk_{chunk_idx}.jsonl"
        with open(chunk_file_path, 'w') as f:
            for req in chunk_requests:
                f.write(json.dumps(req) + '\n')
        
        # Upload file
        print("Uploading chunk file...")
        with open(chunk_file_path, 'rb') as f:
            batch_file = client.files.create(file=f, purpose="batch")
        
        print(f"Uploaded file ID: {batch_file.id}")
        
        # Create batch
        print("Creating batch job...")
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"QEvasion COT chunk {chunk_idx + 1}/{total_chunks}"}
        )
        
        print("Batch created!")
        print(f"  Batch ID: {batch.id}")
        print(f"  Status: {batch.status}")
        
        # Save batch info
        status_info["batches"][str(chunk_idx)] = {
            "batch_id": batch.id,
            "input_file_id": batch_file.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "start_idx": start_idx,
            "end_idx": end_idx
        }
        
        with open(BATCH_STATUS_FILE, 'w') as f:
            json.dump(status_info, f, indent=2)
    
    submitted_count = len(status_info["batches"])
    remaining = total_chunks - submitted_count
    
    print(f"\n📊 Progress: {submitted_count}/{total_chunks} chunks submitted")
    if remaining > 0:
        print(f"⏳ {remaining} chunks remaining")
        print("\nRun 'python generate-cot.py --create' again to submit the next chunk")
        print("(after current batches complete to avoid token limit)")
    
    print("\nRun 'python generate-cot.py --status' to check progress")
    
    return batch


def check_status():
    """Check batch status for all chunks"""
    
    if not Path(BATCH_STATUS_FILE).exists():
        print("No batch found. Run with --create first.")
        return None
    
    with open(BATCH_STATUS_FILE, 'r') as f:
        status_info = json.load(f)
    
    # Handle old format (single batch)
    if "batches" not in status_info:
        batch = client.batches.retrieve(status_info["batch_id"])
        print(f"Batch ID: {batch.id}")
        print(f"Status: {batch.status}")
        print(f"Progress: {batch.request_counts.completed}/{batch.request_counts.total} completed")
        print(f"Failed: {batch.request_counts.failed}")
        
        if batch.status == "completed":
            print("\n✅ Batch completed!")
            print(f"Output file ID: {batch.output_file_id}")
            print("\nRun 'python generate-cot.py --download' to get results")
            status_info["status"] = batch.status
            status_info["output_file_id"] = batch.output_file_id
            with open(BATCH_STATUS_FILE, 'w') as f:
                json.dump(status_info, f, indent=2)
        elif batch.status == "failed":
            print("\n❌ Batch failed!")
            if batch.errors:
                print(f"Errors: {batch.errors}")
        elif batch.status in ["validating", "in_progress", "finalizing"]:
            print("\n⏳ Batch still processing...")
        return batch
    
    # New format (multiple batches)
    total_chunks = status_info.get("total_chunks", len(status_info["batches"]))
    total_requests = status_info.get("total_requests", 0)
    
    print("📊 Batch Status Overview")
    print(f"   Total requests: {total_requests}")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Submitted: {len(status_info['batches'])}/{total_chunks}")
    print("-" * 50)
    
    all_completed = True
    total_completed = 0
    total_failed = 0
    can_submit_more = True
    
    for chunk_idx in sorted(status_info["batches"].keys(), key=int):
        batch_info = status_info["batches"][chunk_idx]
        batch = client.batches.retrieve(batch_info["batch_id"])
        
        # Update status
        batch_info["status"] = batch.status
        if batch.output_file_id:
            batch_info["output_file_id"] = batch.output_file_id
        
        completed = batch.request_counts.completed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0
        failed = batch.request_counts.failed if batch.request_counts else 0
        
        total_completed += completed
        total_failed += failed
        
        status_emoji = {
            "completed": "✅",
            "failed": "❌",
            "in_progress": "🔄",
            "validating": "⏳",
            "finalizing": "📦",
            "cancelling": "🚫",
            "cancelled": "🚫",
            "expired": "⏰"
        }.get(batch.status, "❓")
        
        print(f"  Chunk {int(chunk_idx) + 1}: {status_emoji} {batch.status} ({completed}/{total})")
        
        if batch.status != "completed":
            all_completed = False
        
        if batch.status in ["validating", "in_progress", "finalizing"]:
            can_submit_more = False
        
        if batch.status == "failed" and batch.errors:
            print(f"         Error: {batch.errors}")
    
    # Save updated status
    with open(BATCH_STATUS_FILE, 'w') as f:
        json.dump(status_info, f, indent=2)
    
    print("-" * 50)
    print(f"Total progress: {total_completed} completed, {total_failed} failed")
    
    # Recommendations
    submitted = len(status_info["batches"])
    remaining_chunks = total_chunks - submitted
    
    if all_completed and submitted == total_chunks:
        print("\n✅ All batches completed!")
        print("Run 'python generate-cot.py --download' to merge results")
    elif remaining_chunks > 0:
        if can_submit_more:
            print(f"\n📤 Ready to submit more chunks ({remaining_chunks} remaining)")
            print("Run 'python generate-cot.py --create' to submit next chunk")
        else:
            print("\n⏳ Wait for current batches to complete before submitting more")
            print(f"   ({remaining_chunks} chunks remaining)")
    else:
        print("\n⏳ Waiting for batches to complete...")
    
    return None


def download_results():
    """Download batch results and create training data"""
    
    if not Path(BATCH_STATUS_FILE).exists():
        print("No batch found. Run with --create first.")
        return
    
    with open(BATCH_STATUS_FILE, 'r') as f:
        status_info = json.load(f)
    
    # Handle old format (single batch)
    if "batches" not in status_info:
        if "output_file_id" not in status_info:
            batch = client.batches.retrieve(status_info["batch_id"])
            if batch.status != "completed":
                print(f"Batch not yet completed. Status: {batch.status}")
                return
            status_info["output_file_id"] = batch.output_file_id
        
        print(f"Downloading results from: {status_info['output_file_id']}")
        file_response = client.files.content(status_info["output_file_id"])
        
        output_path = f"{OUTPUT_DIR}/batch_output.jsonl"
        with open(output_path, 'w') as f:
            f.write(file_response.text)
        
        print(f"Downloaded to: {output_path}")
        process_batch_results(output_path)
        return
    
    # New format (multiple batches)
    total_chunks = status_info.get("total_chunks", len(status_info["batches"]))
    submitted = len(status_info["batches"])
    
    if submitted < total_chunks:
        print(f"⚠️  Only {submitted}/{total_chunks} chunks have been submitted.")
        print("Run 'python generate-cot.py --create' to submit remaining chunks first.")
        response = input("Download available results anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Check all batches are completed and download
    all_results = []
    
    for chunk_idx in sorted(status_info["batches"].keys(), key=int):
        batch_info = status_info["batches"][chunk_idx]
        
        # Get latest status if no output file yet
        if "output_file_id" not in batch_info:
            batch = client.batches.retrieve(batch_info["batch_id"])
            if batch.status != "completed":
                print(f"Chunk {int(chunk_idx) + 1} not yet completed. Status: {batch.status}")
                continue
            batch_info["output_file_id"] = batch.output_file_id
        
        # Download this chunk's results
        print(f"Downloading chunk {int(chunk_idx) + 1}...")
        file_response = client.files.content(batch_info["output_file_id"])
        
        chunk_output_path = f"{OUTPUT_DIR}/batch_output_chunk_{chunk_idx}.jsonl"
        with open(chunk_output_path, 'w') as f:
            f.write(file_response.text)
        
        # Collect results
        for line in file_response.text.strip().split('\n'):
            if line:
                all_results.append(line)
    
    if not all_results:
        print("No results available to download yet.")
        return
    
    # Save updated status
    with open(BATCH_STATUS_FILE, 'w') as f:
        json.dump(status_info, f, indent=2)
    
    # Merge all results into single file
    merged_output_path = f"{OUTPUT_DIR}/batch_output.jsonl"
    with open(merged_output_path, 'w') as f:
        for result in all_results:
            f.write(result + '\n')
    
    print(f"\n📥 Downloaded and merged {len(all_results)} results")
    print(f"   Saved to: {merged_output_path}")
    
    # Process merged results
    process_batch_results(merged_output_path)


def process_batch_results(output_path: str):
    """Process batch output and create training data"""
    
    # Load original training data
    train_df = pd.read_parquet(f"{OUTPUT_DIR}/train_original.parquet")
    
    # Parse batch results
    results = {}
    with open(output_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result["custom_id"]
            idx = int(custom_id.replace("request-", ""))
            
            if result["error"]:
                print(f"Error for {custom_id}: {result['error']}")
                continue
            
            # Parse the response
            response_body = result["response"]["body"]
            content = response_body["choices"][0]["message"]["content"]
            
            try:
                parsed = json.loads(content)
                results[idx] = parsed
            except json.JSONDecodeError as e:
                print(f"JSON parse error for {custom_id}: {e}")
                continue
    
    print(f"Successfully parsed {len(results)}/{len(train_df)} responses")
    
    # Create training data
    formatted = []
    
    for idx, row in train_df.iterrows():
        if idx not in results:
            continue
        
        parsed = results[idx]
        
        # Build reasoning from structured steps
        reasoning = f"""Step 1 - What the question asks: {parsed['step1_question_asks']}

Step 2 - How the answer addresses it: {parsed['step2_answer_addresses']}

Step 3 - Information type: {parsed['step3_information_type']}

Step 4 - Evasion check: {parsed['step4_evasion_check']}

Summary: {parsed['reasoning_summary']}"""
        
        # Use ground truth label (reasoning was generated to justify this label)
        raw_label = str(row['clarity_label']).strip()
        label_map = {
            'clear reply': 'Clear Reply',
            'ambivalent': 'Ambivalent', 
            'clear non-reply': 'Clear Non-Reply',
            'clear non reply': 'Clear Non-Reply',
        }
        label = label_map.get(raw_label.lower(), raw_label)
        
        prompt = f"""Based on a segment of the interview where the interviewer asks a series of questions, classify the type of response provided by the interviewee for the following question.

### Classification Categories ###
1. Clear Reply - The information requested is explicitly stated (in the requested form)
2. Clear Non-Reply - The information requested is not given at all due to ignorance, need for clarification, or declining to answer
3. Ambivalent - The information requested is given in an incomplete way (e.g., the answer is too general, partial, implicit, dodging, or deflection)

### Full Interview Question ###
{str(row['interview_question']).strip()}

### Full Interview Answer ###
{str(row['interview_answer']).strip()}

### Specific Question to Classify ###
{str(row['question']).strip()}

Think step by step, then provide your classification."""

        response = f"""{reasoning}

LABEL: {label}"""

        formatted.append({
            "conversations": [
                {
                    "role": "system",
                    "content": "You are an expert political discourse analyst. Analyze political interviews step by step and classify response clarity."
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        })
    
    # Save training data
    train_path = f"{OUTPUT_DIR}/train_cot.jsonl"
    with open(train_path, 'w') as f:
        for item in formatted:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✅ Created training data: {train_path}")
    print(f"   Total examples: {len(formatted)}")
    print("\nLoad in Unsloth:")
    print(f"   dataset = load_dataset('json', data_files='{train_path}', split='train')")


def demo():
    """Show a demo of what gets sent to GPT-4o"""
    
    # Example data with ground truth label
    example = {
        "interview_question": "Q. Of the Biden administration. And accused the United States of containing China while pushing for diplomatic talks. How would you respond to that? And do you think President Xi is being sincere about getting the relationship back on track as he bans Apple in China?",
        "interview_answer": "Well, look, first of all, the—I am sincere about getting the relationship right. And one of the things that is going on now is, China is beginning to change some of the rules of the game, in terms of trade and other issues. And so one of the things we talked about, for example, is that they're now talking about making sure that no Chinese—no one in the Chinese Government can use a Western cell phone. Those kinds of things. And so, really, what this trip was about—it was less about containing China. I don't want to contain China. I just want to make sure that we have a relationship with China that is on the up and up, squared away, everybody knows what it's all about.",
        "question": "How would you respond to the accusation that the United States is containing China while pushing for diplomatic talks?",
        "clarity_label": "Clear Reply"
    }
    
    prompt = create_cot_prompt(
        example["interview_question"],
        example["interview_answer"],
        example["question"],
        example["clarity_label"]
    )
    
    request = {
        "custom_id": "request-0",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-2024-08-06",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert political discourse analyst specializing in classifying response clarity in political interviews. Analyze each question-answer pair step by step."
                },
                {"role": "user", "content": prompt}
            ],
            "response_format": COT_RESPONSE_SCHEMA,
            "temperature": 0.1,
            "max_tokens": 1000
        }
    }
    
    print("=" * 80)
    print("DEMO: What gets sent to GPT-4o (one batch request)")
    print("NOTE: GPT generates reasoning to justify the ground truth label from dataset")
    print("=" * 80)
    
    print("\n>>> SYSTEM MESSAGE:")
    print("-" * 40)
    print(request["body"]["messages"][0]["content"])
    
    print("\n>>> USER MESSAGE (the prompt):")
    print("-" * 40)
    print(request["body"]["messages"][1]["content"])
    
    print("\n>>> RESPONSE FORMAT (JSON Schema - guarantees valid output):")
    print("-" * 40)
    print(json.dumps(COT_RESPONSE_SCHEMA["json_schema"]["schema"]["properties"], indent=2))
    
    print("\n>>> EXPECTED OUTPUT FORMAT (reasoning to justify the ground truth label):")
    print("-" * 40)
    expected = {
        "step1_question_asks": "The question asks how the President would respond to accusations that the US is containing China while seeking diplomatic talks.",
        "step2_answer_addresses": "The President directly addresses this by stating the trip was not about containing China, emphasizing his desire for a transparent relationship.",
        "step3_information_type": "The information is explicit - he directly says 'I don't want to contain China' and explains the actual purpose.",
        "step4_evasion_check": "No evasion detected. The President directly addresses the accusation with a clear denial and explanation.",
        "reasoning_summary": "The classification 'Clear Reply' is correct because the President explicitly denies the containment accusation and provides clear reasoning about the trip's purpose."
    }
    print(json.dumps(expected, indent=2))
    
    print("\n" + "=" * 80)
    print("This request format is repeated for all ~2500 training examples")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate COT using Batch API")
    parser.add_argument("--create", action="store_true", help="Create and submit next batch chunk")
    parser.add_argument("--chunk", type=int, help="Specific chunk index to submit (0-indexed)")
    parser.add_argument("--status", action="store_true", help="Check batch status")
    parser.add_argument("--download", action="store_true", help="Download results")
    parser.add_argument("--demo", action="store_true", help="Show demo of request format")
    parser.add_argument("--reset", action="store_true", help="Reset batch status (start fresh)")
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    elif args.reset:
        if Path(BATCH_STATUS_FILE).exists():
            Path(BATCH_STATUS_FILE).unlink()
            print("✅ Reset batch status. Run --create to start fresh.")
        else:
            print("No batch status to reset.")
    elif args.create:
        submit_batch(chunk_index=args.chunk)
    elif args.status:
        check_status()
    elif args.download:
        download_results()
    else:
        parser.print_help()
        print("\n\nWorkflow (with chunked batches to avoid token limits):")
        print(f"  Batch size: {BATCH_SIZE} requests per chunk")
        print("")
        print("  1. python generate-cot.py --demo       # See what gets sent to GPT")
        print("  2. python generate-cot.py --create     # Submit first chunk")
        print("  3. python generate-cot.py --status     # Check progress")
        print("  4. python generate-cot.py --create     # Submit next chunk (when ready)")
        print("     ... repeat 3-4 until all chunks submitted ...")
        print("  5. python generate-cot.py --download   # Merge & create training data")
        print("")
        print("  Options:")
        print("    --chunk N    Submit specific chunk (0-indexed)")
        print("    --reset      Clear batch status and start fresh")


if __name__ == "__main__":
    main()