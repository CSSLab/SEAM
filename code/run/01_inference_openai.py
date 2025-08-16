#!/usr/bin/env python3
"""
SEAM Benchmark - OpenAI API Inference Only
Pure inference script for OpenAI models, outputs raw results to unified format.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import wraps

# Third-party imports
import openai
from PIL import Image

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Local imports
from config.config import *
from config.config import RESULTS_DIR, BENCHMARK_ROOT
from utils.task_loader import TaskLoader

# Progress bar
from tqdm import tqdm

def normalize_model_name(model_name: str) -> str:
    """Convert model names to filesystem-safe format"""
    return model_name.replace('/', '-').replace('_', '-').lower()

def retry_with_exponential_backoff(
    max_retries: int = 20,
    initial_delay: float = 2.0,
    exponential_base: float = 1.5,
    max_delay: float = 120.0
):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except openai.RateLimitError as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    print(f"⏳ Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
                except openai.APITimeoutError as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    print(f"⏳ Timeout, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
                except openai.APIConnectionError as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    print(f"⏳ Connection error, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
                except Exception as e:
                    # For other exceptions, don't retry
                    raise e
            
            # If we get here, all retries failed
            raise last_exception
        return wrapper
    return decorator

def load_api_key():
    """Load OpenAI API key from config or environment"""
    try:
        # Try loading from JSON config first
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'api_keys.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_key = config.get('openai', {}).get('api_key')
                if api_key and api_key != "your_openai_api_key_here":
                    return api_key
    except:
        pass
    
    # Fallback to environment variable
    return os.getenv('OPENAI_API_KEY')

def encode_image(image_path: str) -> str:
    """Encode image to base64 for OpenAI API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class OpenAIInferenceRunner:
    """Pure OpenAI API inference runner - no answer extraction"""
    
    def __init__(self, model_name: str, max_tokens: int = None):
        self.model_name = model_name
        self.normalized_name = normalize_model_name(model_name)
        self.max_tokens = max_tokens or MODEL_MAX_TOKENS
        
        print(f"🚀 Initializing OpenAI Inference Runner")
        print(f"   Model: {model_name}")
        print(f"   Normalized: {self.normalized_name}")
        print(f"   Temperature: model_default")
        print(f"   Max tokens: {self.max_tokens}")
        
        # Setup API key
        api_key = load_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or add to api_keys.json")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        # Setup results directory
        self.results_dir = RESULTS_DIR / self.normalized_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.results_dir / "output.jsonl"
        
        print(f"✅ OpenAI client initialized!")
        print(f"   Output: {self.output_file}")
    
    def run_inference(self, benchmark_root: str, modes: List[str], tasks: List[str] = None,
                     debug_samples: int = 0, resume: bool = True, parallel: bool = True, max_workers: int = 10) -> int:
        """Run inference across specified modes and tasks"""
        
        task_loader = TaskLoader(benchmark_root)
        all_tasks = tasks if tasks else task_loader.all_tasks
        
        print(f"\n📊 Starting SEAM Inference")
        print(f"   Tasks: {len(all_tasks)} ({', '.join(all_tasks)})")
        print(f"   Modes: {', '.join(modes)}")
        print(f"   Debug samples: {debug_samples if debug_samples > 0 else 'All'}")
        print(f"   Resume: {resume}")
        print(f"   Parallel: {parallel} (max_workers={max_workers if parallel else 'N/A'})")
        
        # Load existing results for resume functionality
        existing_results = set()
        if resume and self.output_file.exists():
            try:
                with open(self.output_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line.strip())
                            key = f"{result['task_name']}_{result['mode']}_{result['index']}"
                            existing_results.add(key)
                print(f"   Found {len(existing_results)} existing results")
            except Exception as e:
                print(f"   Warning: Could not load existing results: {e}")
        
        total_processed = 0
        start_time = time.time()
        
        # Process each mode sequentially (API has rate limits)
        for mode in modes:
            print(f"\n🔄 Processing mode: {mode.upper()}")
            
            prompts = self.generate_prompts(task_loader, all_tasks, mode, debug_samples, existing_results)
            if not prompts:
                print(f"   No new prompts for mode {mode}")
                continue
            
            # Choose processing method
            if parallel:
                mode_results = self.process_prompts_parallel(prompts, mode, task_loader, max_workers)
            else:
                mode_results = self.process_prompts(prompts, mode, task_loader)
            
            total_processed += len(mode_results)
            
            print(f"   Processed {len(mode_results)} samples for mode {mode}")
        
        total_time = time.time() - start_time
        throughput = total_processed / total_time if total_time > 0 else 0
        
        print(f"\n✅ Inference Completed!")
        print(f"   Total processed: {total_processed}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Throughput: {throughput:.2f} samples/sec")
        print(f"   Results saved: {self.output_file}")
        
        return total_processed
    
    def generate_prompts(self, task_loader: TaskLoader, tasks: List[str], 
                        mode: str, debug_samples: int, existing_results: set) -> List[dict]:
        """Generate prompts for specified tasks and mode"""
        prompts = []
        
        for task_name in tasks:
            try:
                task_data = task_loader.load_task_data(task_name)
                
                # Limit samples if debugging
                if debug_samples > 0:
                    task_data = task_data[:debug_samples]
                
                for idx, sample in enumerate(task_data):
                    # Check if already processed
                    key = f"{task_name}_{mode}_{idx}"
                    if key in existing_results:
                        continue
                        
                    try:
                        # Get formatted prompt from task loader
                        prompt_text, answer, notation = task_loader.format_task(task_name, sample, mode)
                        
                        prompt_data = {
                            "model": self.normalized_name,
                            "task_name": task_name,
                            "mode": mode,
                            "index": idx,
                            "question": prompt_text,
                            "answer": answer,
                            "sample": sample  # Keep original sample for image paths
                        }
                        
                        prompts.append(prompt_data)
                        
                    except Exception as e:
                        print(f"⚠️  Error formatting {task_name}[{idx}]: {e}")
                        continue
                        
            except Exception as e:
                print(f"⚠️  Error loading task {task_name}: {e}")
                continue
            
        print(f"📝 Generated {len(prompts)} new prompts for mode '{mode}'")
        return prompts
    
    def process_prompts(self, prompts: List[dict], mode: str, task_loader: TaskLoader) -> List[dict]:
        """Process prompts through OpenAI API with appropriate formatting"""
        results = []
        
        progress_bar = tqdm(prompts, desc=f"Processing {mode} prompts")
        
        for prompt_data in progress_bar:
            try:
                start_time = time.time()
                
                # Build message based on mode
                if mode == 'l':  # Language only
                    messages = [
                        {"role": "system", "content": MODEL_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_data["question"]}
                    ]
                elif mode in ['v', 'vl']:  # Vision modes
                    messages = self.build_vision_messages(prompt_data, mode, task_loader)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=self.max_tokens
                )
                
                
                # Extract response
                output_text = response.choices[0].message.content
                
                # Build result
                result = {
                    "model": prompt_data["model"],
                    "task_name": prompt_data["task_name"],
                    "mode": prompt_data["mode"],
                    "index": prompt_data["index"],
                    "question": prompt_data["question"],
                    "answer": prompt_data["answer"],
                    "output": output_text,
                }
                
                results.append(result)
                
                # Save incrementally
                self.save_result(result)
                
            except Exception as e:
                print(f"⚠️  Error processing {prompt_data['task_name']}[{prompt_data['index']}]: {e}")
                continue
            
        return results
    
    @retry_with_exponential_backoff(max_retries=20, initial_delay=2.0)
    def process_single_prompt(self, prompt_data: dict, mode: str, task_loader: TaskLoader) -> dict:
        """Process a single prompt with retry logic"""
        start_time = time.time()
        
        # Build message based on mode
        if mode == 'l':  # Language only
            messages = [
                {"role": "system", "content": MODEL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_data["question"]}
            ]
        elif mode in ['v', 'vl']:  # Vision modes
            messages = self.build_vision_messages(prompt_data, mode, task_loader)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Call OpenAI API with retry
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=self.max_tokens
        )
        
        # Extract response
        output_text = response.choices[0].message.content
        processing_time = time.time() - start_time
        
        # Build result
        result = {
            **prompt_data,  # Include all original fields
            "output": output_text,
            "processing_time": processing_time
        }
        
        return result
    
    def process_prompts_parallel(self, prompts: List[dict], mode: str, task_loader: TaskLoader, max_workers: int = 10) -> List[dict]:
        """Process prompts in parallel with retry logic and guaranteed completion"""
        results = []
        remaining_prompts = prompts.copy()
        retry_round = 0
        max_retry_rounds = 5
        
        print(f"🚀 Processing {len(prompts)} prompts in parallel (max_workers={max_workers})")
        print(f"   Will retry failed prompts up to {max_retry_rounds} rounds until ALL are completed")
        
        while remaining_prompts and retry_round < max_retry_rounds:
            if retry_round > 0:
                print(f"\n🔄 Retry round {retry_round}: {len(remaining_prompts)} prompts remaining")
                # Reduce workers and increase delays on retries for better stability
                current_workers = max(1, max_workers // (retry_round + 1))
                print(f"   Using {current_workers} workers for retry stability")
            else:
                current_workers = max_workers
            
            round_results = []
            failed_prompts = []
            
            with ThreadPoolExecutor(max_workers=current_workers) as executor:
                # Submit current batch of prompts
                future_to_prompt = {
                    executor.submit(self.process_single_prompt, prompt_data, mode, task_loader): prompt_data
                    for prompt_data in remaining_prompts
                }
                
                # Process completed tasks with progress bar
                progress_desc = f"Processing {mode} prompts" if retry_round == 0 else f"Retry {retry_round} - {mode} prompts"
                progress_bar = tqdm(total=len(remaining_prompts), desc=progress_desc)
                batch_results = []
                
                for future in as_completed(future_to_prompt):
                    prompt_data = future_to_prompt[future]
                    try:
                        result = future.result()
                        round_results.append(result)
                        batch_results.append(result)
                        
                        # Save in batches of 50 for incremental progress
                        if len(batch_results) >= 50:
                            self.save_results(batch_results)
                            batch_results = []
                            
                    except Exception as e:
                        print(f"⚠️  Failed: {prompt_data['task_name']}[{prompt_data['index']}] - {e}")
                        failed_prompts.append(prompt_data)
                    
                    progress_bar.update(1)
                
                # Save any remaining results from this round
                if batch_results:
                    self.save_results(batch_results)
                
                progress_bar.close()
            
            # Add successful results to overall results
            results.extend(round_results)
            
            # Update remaining prompts for next round
            remaining_prompts = failed_prompts
            retry_round += 1
            
            if remaining_prompts:
                wait_time = min(30 * retry_round, 120)  # Exponential wait between rounds, max 2 minutes
                print(f"   ⏳ Waiting {wait_time}s before retry round {retry_round}...")
                time.sleep(wait_time)
        
        # Final status report
        if remaining_prompts:
            print(f"\n❌ CRITICAL: {len(remaining_prompts)} prompts could not be processed after {max_retry_rounds} retry rounds!")
            for prompt in remaining_prompts[:5]:  # Show first 5 failed prompts
                print(f"   - {prompt['task_name']}[{prompt['index']}]")
            if len(remaining_prompts) > 5:
                print(f"   - ... and {len(remaining_prompts) - 5} more")
        else:
            print(f"\n🎉 SUCCESS: ALL {len(prompts)} prompts processed successfully!")
        
        print(f"✅ Final result: {len(results)}/{len(prompts)} prompts completed")
        return results
    
    def save_results(self, results: List[dict]):
        """Save results to output file"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
    
    def build_vision_messages(self, prompt_data: dict, mode: str, task_loader: TaskLoader) -> List[dict]:
        """Build messages for vision-enabled modes"""
        if mode == 'v':
            # Vision only - use image without text question
            image_path = task_loader.get_image_path(prompt_data["task_name"], prompt_data["sample"])
            if not image_path or not os.path.exists(image_path):
                raise ValueError(f"Image not found: {image_path}")
            
            # Create vision-only prompt
            vision_prompt = task_loader.get_vision_only_prompt(prompt_data["task_name"], prompt_data["sample"])
            
            base64_image = encode_image(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        elif mode == 'vl':
            # Vision + Language - include both image and text
            image_path = task_loader.get_image_path(prompt_data["task_name"], prompt_data["sample"])
            if not image_path or not os.path.exists(image_path):
                raise ValueError(f"Image not found: {image_path}")
            
            base64_image = encode_image(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_data["question"]},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        else:
            raise ValueError(f"Unknown vision mode: {mode}")
        
        return messages
    
    def save_result(self, result: dict):
        """Append single result to output file"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            # Remove sample field before saving
            clean_result = {k: v for k, v in result.items() if k != 'sample'}
            json.dump(clean_result, f, ensure_ascii=False)
            f.write('\n')

class OpenAIBatchRunner:
    """OpenAI Batch API runner for large-scale processing"""
    
    def __init__(self, model_name: str, max_tokens: int = None):
        self.model_name = model_name
        self.normalized_name = normalize_model_name(model_name)
        self.max_tokens = max_tokens or MODEL_MAX_TOKENS
        
        # Setup API key
        api_key = load_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or add to api_keys.json")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        # Setup results directory
        self.results_dir = RESULTS_DIR / self.normalized_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.batch_dir = self.results_dir / "batch"
        self.batch_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Initializing OpenAI Batch Runner")
        print(f"   Model: {model_name}")
        print(f"   Batch directory: {self.batch_dir}")
    
    def prepare_batch_data(self, benchmark_root: str, modes: List[str], tasks: List[str] = None) -> List[str]:
        """Prepare batch data files for OpenAI batch API (one file per mode)"""
        task_loader = TaskLoader(benchmark_root)
        all_tasks = tasks if tasks else task_loader.all_tasks
        
        print(f"📝 Preparing batch data files (split by mode)...")
        
        batch_files = []
        
        for mode in modes:
            batch_file = self.batch_dir / f"batch_requests_{mode}.jsonl"
            batch_files.append(str(batch_file))
            
            print(f"  📄 Creating batch file for mode '{mode}': {batch_file.name}")
            
            request_count = 0
            with open(batch_file, 'w', encoding='utf-8') as f:
                for task_name in all_tasks:
                    try:
                        task_data = task_loader.load_task_data(task_name)
                        
                        for idx, sample in enumerate(task_data):
                            try:
                                prompt_text, answer, notation = task_loader.format_task(task_name, sample, mode)
                                
                                # Build request based on mode
                                if mode == 'l':
                                    messages = [
                                        {"role": "system", "content": MODEL_SYSTEM_PROMPT},
                                        {"role": "user", "content": prompt_text}
                                    ]
                                elif mode in ['v', 'vl']:
                                    # For batch API, we need to handle images differently
                                    image_path = task_loader.get_image_path(task_name, sample)
                                    if image_path and os.path.exists(image_path):
                                        base64_image = encode_image(image_path)
                                        
                                        if mode == 'v':
                                            vision_prompt = task_loader.get_vision_only_prompt(task_name, sample)
                                            content = [
                                                {"type": "text", "text": vision_prompt},
                                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                            ]
                                        else:  # vl
                                            content = [
                                                {"type": "text", "text": prompt_text},
                                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                            ]
                                        
                                        messages = [{"role": "user", "content": content}]
                                    else:
                                        print(f"⚠️  Image not found for {task_name}[{idx}], skipping")
                                        continue
                                
                                batch_request = {
                                    "custom_id": f"{task_name}_{mode}_{idx}",
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": {
                                        "model": self.model_name,
                                        "messages": messages,
                                        "max_completion_tokens": self.max_tokens
                                    }
                                }
                                
                                f.write(json.dumps(batch_request) + '\n')
                                request_count += 1
                                
                            except Exception as e:
                                print(f"⚠️  Error preparing {task_name}[{idx}]: {e}")
                                continue
                                
                    except Exception as e:
                        print(f"⚠️  Error loading task {task_name}: {e}")
                        continue
            
            print(f"  ✅ Mode '{mode}': {request_count} requests in {batch_file.name}")
        
        total_requests = sum(len(open(bf).readlines()) for bf in batch_files)
        print(f"📊 Total prepared: {total_requests} requests across {len(batch_files)} files")
        
        return batch_files
    
    def save_batch_ids(self, batch_ids: Dict[str, str]):
        """Save batch IDs to file for tracking"""
        batch_ids_file = self.batch_dir / "batch_ids.json"
        with open(batch_ids_file, 'w') as f:
            json.dump(batch_ids, f, indent=2)
        print(f"💾 Batch IDs saved to: {batch_ids_file}")
    
    def load_batch_ids(self) -> Dict[str, str]:
        """Load batch IDs from file"""
        batch_ids_file = self.batch_dir / "batch_ids.json"
        if batch_ids_file.exists():
            with open(batch_ids_file, 'r') as f:
                return json.load(f)
        return {}
    
    def upload_batch_file(self, batch_file: str) -> str:
        """Upload batch file to OpenAI"""
        print(f"📤 Uploading batch file...")
        
        with open(batch_file, "rb") as f:
            batch_input_file = self.client.files.create(
                file=f,
                purpose="batch"
            )
        
        print(f"✅ File uploaded: {batch_input_file.id}")
        return batch_input_file.id
    
    def submit_batch(self, input_file_id: str) -> str:
        """Submit batch job to OpenAI"""
        print(f"🚀 Submitting batch job...")
        
        batch_job = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"SEAM benchmark evaluation for {self.model_name}"}
        )
        
        print(f"✅ Batch job submitted: {batch_job.id}")
        print(f"   Status: {batch_job.status}")
        
        # Save batch info
        batch_info_file = self.batch_dir / "batch_info.json"
        with open(batch_info_file, 'w') as f:
            json.dump({
                "batch_id": batch_job.id,
                "input_file_id": input_file_id,
                "model": self.model_name,
                "status": batch_job.status,
                "created_at": batch_job.created_at,
                "metadata": batch_job.metadata
            }, f, indent=2)
        
        return batch_job.id
    
    def check_batch_status(self, batch_id: str = None) -> dict:
        """Check status of batch job"""
        if not batch_id:
            # Try to load from batch info
            batch_info_file = self.batch_dir / "batch_info.json"
            if batch_info_file.exists():
                with open(batch_info_file, 'r') as f:
                    batch_info = json.load(f)
                    batch_id = batch_info.get("batch_id")
        
        if not batch_id:
            raise ValueError("No batch ID provided or found")
        
        batch_job = self.client.batches.retrieve(batch_id)
        
        print(f"📊 Batch Status: {batch_job.status}")
        print(f"   Progress: {batch_job.request_counts}")
        
        return {
            "status": batch_job.status,
            "request_counts": batch_job.request_counts,
            "output_file_id": batch_job.output_file_id,
            "error_file_id": batch_job.error_file_id
        }
    
    def download_results(self, batch_id: str = None) -> str:
        """Download and convert batch results"""
        batch_status = self.check_batch_status(batch_id)
        
        if batch_status["status"] != "completed":
            print(f"❌ Batch not completed yet. Status: {batch_status['status']}")
            return None
        
        output_file_id = batch_status["output_file_id"]
        if not output_file_id:
            print(f"❌ No output file available")
            return None
        
        print(f"📥 Downloading results...")
        
        # Download results
        file_response = self.client.files.content(output_file_id)
        
        # Save raw results
        raw_results_file = self.batch_dir / "raw_results.jsonl"
        with open(raw_results_file, 'wb') as f:
            f.write(file_response.content)
        
        # Convert to unified format
        output_file = self.convert_batch_results(str(raw_results_file))
        
        print(f"✅ Results downloaded and converted")
        print(f"   Output: {output_file}")
        
        return output_file
    
    def download_and_merge_results(self) -> str:
        """Download results from all batches and merge into single output file"""
        batch_ids = self.load_batch_ids()
        if not batch_ids:
            raise FileNotFoundError("No batch IDs found. Run submit first.")
        
        output_file = self.results_dir / "output.jsonl"
        all_results = []
        
        print(f"📥 Downloading and merging results from {len(batch_ids)} batches...")
        
        for mode, batch_id in batch_ids.items():
            print(f"  📦 Downloading mode '{mode}' results...")
            
            # Download individual batch results to mode-specific file
            mode_output = self.download_single_batch(batch_id, mode)
            
            # Read and collect results
            with open(mode_output, 'r') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line.strip())
                        all_results.append(result)
        
        # Write merged results
        print(f"💾 Merging {len(all_results)} results into {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in all_results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ Merged results saved: {output_file}")
        return str(output_file)
    
    def download_single_batch(self, batch_id: str, mode: str) -> str:
        """Download results for a single batch"""
        batch_status = self.check_batch_status(batch_id)
        
        if batch_status["status"] != "completed":
            raise ValueError(f"Batch {batch_id} not completed. Status: {batch_status['status']}")
        
        output_file_id = batch_status["output_file_id"]
        if not output_file_id:
            raise ValueError(f"No output file available for batch {batch_id}")
        
        # Download results
        file_response = self.client.files.content(output_file_id)
        
        # Save raw results with mode suffix
        raw_results_file = self.batch_dir / f"raw_results_{mode}.jsonl"
        with open(raw_results_file, 'wb') as f:
            f.write(file_response.content)
        
        # Convert to unified format with mode suffix
        output_file = self.convert_batch_results(str(raw_results_file), mode)
        
        return output_file
    
    def convert_batch_results(self, raw_results_file: str, mode: str = None) -> str:
        """Convert batch results to unified format"""
        print(f"🔄 Converting batch results to unified format...")
        
        # Load task data for ground truth
        task_loader = TaskLoader(str(BENCHMARK_ROOT))
        
        if mode:
            output_file = self.results_dir / f"output_{mode}.jsonl"
        else:
            output_file = self.results_dir / "output.jsonl"
        converted_count = 0
        
        with open(raw_results_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                try:
                    batch_result = json.loads(line.strip())
                    
                    if batch_result.get("response", {}).get("status_code") != 200:
                        print(f"⚠️  Skipping failed request: {batch_result.get('custom_id')}")
                        continue
                    
                    custom_id = batch_result["custom_id"]
                    parts = custom_id.split("_")
                    task_name = "_".join(parts[:-2])
                    mode = parts[-2]
                    index = int(parts[-1])
                    
                    # Get ground truth
                    task_data = task_loader.load_task_data(task_name)
                    sample = task_data[index]
                    _, answer, notation = task_loader.format_task(task_name, sample, mode)
                    
                    # Extract response
                    response_message = batch_result["response"]["body"]["choices"][0]["message"]
                    output_text = response_message["content"]
                    
                    # Build unified result
                    result = {
                        "model": self.normalized_name,
                        "task_name": task_name,
                        "mode": mode,
                        "index": index,
                        "question": "batch_processed",  # Original question not stored in batch
                        "answer": answer,
                        "output": output_text,
                    }
                    
                    json.dump(result, f_out, ensure_ascii=False)
                    f_out.write('\n')
                    converted_count += 1
                    
                except Exception as e:
                    print(f"⚠️  Error converting result: {e}")
                    continue
        
        print(f"✅ Converted {converted_count} results")
        return str(output_file)

def parse_task_list(task_string: str) -> List[str]:
    """Parse comma-separated task list"""
    if not task_string:
        return None
    return [task.strip() for task in task_string.split(',')]

def parse_mode_list(mode_string: str) -> List[str]:
    """Parse comma-separated mode list"""
    modes = [mode.strip() for mode in mode_string.split(',')]
    valid_modes = ['l', 'v', 'vl']
    for mode in modes:
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes: {valid_modes}")
    return modes

def main():
    parser = argparse.ArgumentParser(
        description='SEAM Benchmark OpenAI API Inference (Real-time and Batch)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time inference
  python 01_inference_openai.py --model gpt-4o-mini --modes l,v,vl
  
  # Batch processing
  python 01_inference_openai.py --model gpt-4o-mini --batch --action prepare --modes l,v,vl
  python 01_inference_openai.py --model gpt-4o-mini --batch --action submit
  python 01_inference_openai.py --model gpt-4o-mini --batch --action status
  python 01_inference_openai.py --model gpt-4o-mini --batch --action download
  
  # Full batch pipeline
  python 01_inference_openai.py --model gpt-4o-mini --batch --action all --modes l,v,vl
        """
    )
    
    # Model configuration
    parser.add_argument('--model', required=True, 
                       help='OpenAI model name (e.g., gpt-4o-mini, gpt-4o)')
    parser.add_argument('--max-tokens', type=int, default=None,
                       help=f'Max tokens for generation (default: {MODEL_MAX_TOKENS})')
    
    # Task configuration
    parser.add_argument('--modes', default='l', 
                       help='Comma-separated modes: l,v,vl (default: l)')
    parser.add_argument('--tasks', default=None,
                       help='Comma-separated task names (default: all tasks)')
    
    # Processing configuration
    parser.add_argument('--debug-samples', type=int, default=0,
                       help='Limit samples for debugging (0 for all)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Disable resume functionality')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Enable parallel processing (default: True)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing (use sequential)')
    parser.add_argument('--max-workers', type=int, default=10,
                       help='Maximum number of parallel workers (default: 10)')
    
    # Batch processing configuration
    parser.add_argument('--batch', action='store_true',
                       help='Use batch API instead of real-time')
    parser.add_argument('--action', choices=['prepare', 'submit', 'status', 'download', 'all'],
                       help='Batch action to perform')
    parser.add_argument('--batch-id', help='Specific batch ID for status/download')
    
    args = parser.parse_args()
    
    try:
        # Parse modes and tasks
        modes = parse_mode_list(args.modes)
        tasks = parse_task_list(args.tasks)
        
        # Handle parallel processing arguments
        parallel = args.parallel and not args.no_parallel
        
        if args.batch:
            # Batch processing mode
            if not args.action:
                raise ValueError("--action required for batch processing")
            
            batch_runner = OpenAIBatchRunner(
                model_name=args.model,
                max_tokens=args.max_tokens
            )
            
            if args.action == 'prepare':
                batch_files = batch_runner.prepare_batch_data(
                    benchmark_root=str(BENCHMARK_ROOT),
                    modes=modes,
                    tasks=tasks
                )
                print(f"\n✅ Batch data prepared!")
                print(f"   Files: {[Path(bf).name for bf in batch_files]}")
                print(f"   Next step: python 01_inference_openai.py --model {args.model} --batch --action submit")
                
            elif args.action == 'submit':
                # Find all batch files
                batch_files = list(batch_runner.batch_dir.glob("batch_requests_*.jsonl"))
                if not batch_files:
                    raise FileNotFoundError(f"No batch files found. Run --action prepare first.")
                
                batch_ids = {}
                for batch_file in batch_files:
                    mode = batch_file.stem.split('_')[-1]  # Extract mode from filename
                    print(f"📤 Submitting batch for mode '{mode}'...")
                    
                    input_file_id = batch_runner.upload_batch_file(str(batch_file))
                    batch_id = batch_runner.submit_batch(input_file_id)
                    batch_ids[mode] = batch_id
                    
                    print(f"   ✅ Mode '{mode}' submitted: {batch_id}")
                
                batch_runner.save_batch_ids(batch_ids)
                
                print(f"\n🎉 All batch jobs submitted!")
                print(f"   Modes: {list(batch_ids.keys())}")
                print(f"   Check status: python 01_inference_openai.py --model {args.model} --batch --action status")
                
            elif args.action == 'status':
                batch_ids = batch_runner.load_batch_ids()
                if not batch_ids:
                    print("❌ No batch IDs found. Run --action submit first.")
                    return
                
                print(f"\n📊 Checking status for {len(batch_ids)} batches...")
                all_completed = True
                
                for mode, batch_id in batch_ids.items():
                    status = batch_runner.check_batch_status(batch_id)
                    print(f"   Mode '{mode}': {status['status']} (ID: {batch_id})")
                    if status['status'] != 'completed':
                        all_completed = False
                
                if all_completed:
                    print(f"\n✅ All batches completed!")
                    print(f"   Ready for download: python 01_inference_openai.py --model {args.model} --batch --action download")
                else:
                    print(f"\n⏳ Some batches still processing...")
                
            elif args.action == 'download':
                try:
                    output_file = batch_runner.download_and_merge_results()
                    print(f"\n✅ All batch processing completed!")
                    print(f"   Merged results: {output_file}")
                    print(f"   Next step: python 02_extract.py --model {batch_runner.normalized_name}")
                except Exception as e:
                    print(f"❌ Download failed: {e}")
                    print("   Check batch status first: python 01_inference_openai.py --model {args.model} --batch --action status")
                
            elif args.action == 'all':
                # Full pipeline
                print(f"🚀 Running full batch pipeline...")
                
                # Prepare
                batch_files = batch_runner.prepare_batch_data(
                    benchmark_root=str(BENCHMARK_ROOT),
                    modes=modes,
                    tasks=tasks
                )
                
                # Submit all batches
                batch_ids = {}
                for batch_file in batch_files:
                    mode = Path(batch_file).stem.split('_')[-1]
                    input_file_id = batch_runner.upload_batch_file(batch_file)
                    batch_id = batch_runner.submit_batch(input_file_id)
                    batch_ids[mode] = batch_id
                
                batch_runner.save_batch_ids(batch_ids)
                
                print(f"\n✅ Batch pipeline initiated!")
                print(f"   Submitted {len(batch_ids)} batches: {list(batch_ids.keys())}")
                print(f"   The batches will process in the background.")
                print(f"   Check status: python 01_inference_openai.py --model {args.model} --batch --action status")
                print(f"   Download when ready: python 01_inference_openai.py --model {args.model} --batch --action download")
        
        else:
            # Real-time processing mode
            runner = OpenAIInferenceRunner(
                model_name=args.model,
                max_tokens=args.max_tokens
            )
            
            # Run inference
            total_processed = runner.run_inference(
                benchmark_root=str(BENCHMARK_ROOT),
                modes=modes,
                tasks=tasks,
                debug_samples=args.debug_samples,
                resume=not args.no_resume,
                parallel=parallel,
                max_workers=args.max_workers
            )
            
            print(f"\n🎉 Real-time Inference Completed!")
            print(f"   Model: {args.model}")
            print(f"   Processed: {total_processed} samples")
            print(f"   Next step: python 02_extract.py --model {runner.normalized_name}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()