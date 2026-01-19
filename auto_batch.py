#!/usr/bin/env python3
"""
Automated batch processing script.
Submits chunks and waits for completion, then submits more.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = "./cot_data"
BATCH_STATUS_FILE = f"{OUTPUT_DIR}/batch_status.json"
CHECK_INTERVAL = 60  # seconds

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def log(msg: str):
    """Print with timestamp"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_status():
    """Get current batch status"""
    if not Path(BATCH_STATUS_FILE).exists():
        return None
    with open(BATCH_STATUS_FILE, 'r') as f:
        return json.load(f)


def check_in_progress_batches(status_info: dict) -> tuple[int, int, bool]:
    """Check status of all batches. Returns (completed, in_progress, any_failed)"""
    completed = 0
    in_progress = 0
    any_failed = False
    
    for chunk_idx, batch_info in status_info.get("batches", {}).items():
        batch = client.batches.retrieve(batch_info["batch_id"])
        batch_info["status"] = batch.status
        
        if batch.output_file_id:
            batch_info["output_file_id"] = batch.output_file_id
        
        if batch.status == "completed":
            completed += 1
        elif batch.status in ["validating", "in_progress", "finalizing"]:
            in_progress += 1
        elif batch.status == "failed":
            any_failed = True
            log(f"  ❌ Chunk {int(chunk_idx) + 1} failed!")
    
    # Save updated status
    with open(BATCH_STATUS_FILE, 'w') as f:
        json.dump(status_info, f, indent=2)
    
    return completed, in_progress, any_failed


def submit_next_chunk():
    """Submit the next chunk using generate-cot.py"""
    import subprocess
    result = subprocess.run(
        ["uv", "run", "python", "generate-cot.py", "--create"],
        capture_output=True,
        text=True
    )
    return result.returncode == 0, result.stdout + result.stderr


def main():
    log("🚀 Starting automated batch processing")
    log(f"   Check interval: {CHECK_INTERVAL} seconds")
    print("-" * 50, flush=True)
    
    while True:
        status_info = get_status()
        if not status_info:
            log("No status file found. Run --create first.")
            break
        
        total_chunks = status_info.get("total_chunks", 0)
        submitted = len(status_info.get("batches", {}))
        
        # Check current batch statuses
        log(f"📊 Checking status... ({submitted}/{total_chunks} submitted)")
        completed, in_progress, any_failed = check_in_progress_batches(status_info)
        
        log(f"   ✅ Completed: {completed}/{submitted}")
        if in_progress > 0:
            log(f"   🔄 In progress: {in_progress}")
        
        # If all submitted batches are done
        if completed == submitted:
            remaining = total_chunks - submitted
            
            if remaining == 0:
                log("🎉 All chunks completed!")
                log("Run: python generate-cot.py --download")
                break
            
            # Submit next chunk
            log(f"📤 Submitting chunk {submitted + 1}/{total_chunks}...")
            success, output = submit_next_chunk()
            
            if success:
                # Extract batch ID from output for logging
                if "Batch ID:" in output:
                    batch_id = output.split("Batch ID:")[1].split()[0].strip()
                    log(f"   ✅ Submitted! Batch ID: {batch_id}")
                else:
                    log(f"   ✅ Submitted!")
            else:
                log(f"   ❌ Failed to submit chunk")
                log(f"   Output: {output[:200]}")
                # Wait and retry
                log(f"   Waiting {CHECK_INTERVAL}s before retry...")
        else:
            log(f"   ⏳ Waiting for {submitted - completed} batches to complete...")
        
        # Wait before next check
        log(f"💤 Sleeping {CHECK_INTERVAL}s...")
        print("-" * 50, flush=True)
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
