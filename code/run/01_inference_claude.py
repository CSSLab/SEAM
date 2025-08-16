#!/usr/bin/env python3
"""
SEAM Benchmark - Claude API Inference Only
Pure inference script for Claude models, outputs raw results to unified format.
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
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Third-party imports
import anthropic

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
                except anthropic.RateLimitError as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    print(f"⏳ Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
                except anthropic.APITimeoutError as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    print(f"⏳ Timeout, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
                except anthropic.APIConnectionError as e:
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
    """Load Claude API key from config or environment"""
    try:
        # Try loading from JSON config first
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'api_keys.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get('anthropic', {}).get('api_key')
    except:
        pass
    
    # Fallback to environment variable
    return os.getenv('ANTHROPIC_API_KEY')

def encode_image(image_path: str) -> str:
    """Encode image to base64 for Claude API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_media_type(image_path: str) -> str:
    """Get media type for image file"""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    elif ext == '.png':
        return 'image/png'
    elif ext == '.gif':
        return 'image/gif'
    elif ext == '.webp':
        return 'image/webp'
    else:
        return 'image/jpeg'  # Default

class ClaudeInferenceRunner:
    """Pure Claude API inference runner - no answer extraction"""
    
    def __init__(self, model_name: str, max_tokens: int = None):
        self.model_name = model_name
        self.normalized_name = normalize_model_name(model_name)
        self.max_tokens = max_tokens or MODEL_MAX_TOKENS
        
        print(f"🚀 Initializing Claude Inference Runner")
        print(f"   Model: {model_name}")
        print(f"   Normalized: {self.normalized_name}")
        print(f"   Temperature: model_default")
        print(f"   Max tokens: {self.max_tokens}")
        
        # Setup API key
        api_key = load_api_key()
        if not api_key:
            raise ValueError("Claude API key not found. Set ANTHROPIC_API_KEY environment variable or add to api_keys.json")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Setup results directory
        self.results_dir = RESULTS_DIR / self.normalized_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.results_dir / "output.jsonl"
        
        print(f"✅ Claude client initialized!")
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
                        prompt_text, answer, _ = task_loader.format_task(task_name, sample, mode)
                        
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
    
    @retry_with_exponential_backoff(max_retries=20, initial_delay=2.0)
    def process_single_prompt(self, prompt_data: dict, mode: str, task_loader: TaskLoader) -> dict:
        """Process a single prompt with retry logic"""
        start_time = time.time()
        
        # Build message based on mode
        if mode == 'l':  # Language only
            messages = [
                {
                    "role": "user",
                    "content": prompt_data["question"]
                }
            ]
        elif mode in ['v', 'vl']:  # Vision modes
            messages = self.build_vision_messages(prompt_data, mode, task_loader)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Call Claude API with retry
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            system=MODEL_SYSTEM_PROMPT,
            messages=messages
        )
        
        # Extract response
        output_text = ""
        for content in response.content:
            if content.type == "text":
                output_text += content.text
        
        processing_time = time.time() - start_time
        
        # Build result
        result = {
            **prompt_data,  # Include all original fields
            "output": output_text,
            "processing_time": processing_time
        }
        
        return result
    
    def process_prompts(self, prompts: List[dict], mode: str, task_loader: TaskLoader) -> List[dict]:
        """Process prompts through Claude API with retry logic"""
        results = []
        
        progress_bar = tqdm(prompts, desc=f"Processing {mode} prompts")
        
        for prompt_data in progress_bar:
            try:
                # Use the retry-enabled single prompt method
                result = self.process_single_prompt(prompt_data, mode, task_loader)
                results.append(result)
                
                # Save incrementally
                self.save_result(result)
                
            except Exception as e:
                print(f"⚠️  Error processing {prompt_data['task_name']}[{prompt_data['index']}]: {e}")
                continue
            
        return results
    
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
            media_type = get_image_media_type(image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": vision_prompt
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
            media_type = get_image_media_type(image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt_data["question"]
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
    
    def save_results(self, results: List[dict]):
        """Save results to output file"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            for result in results:
                # Remove sample field before saving
                clean_result = {k: v for k, v in result.items() if k != 'sample'}
                json.dump(clean_result, f, ensure_ascii=False)
                f.write('\n')

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
        description='SEAM Benchmark Claude API Inference Only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference for all tasks and modes (parallel by default)
  python 01_inference_claude.py --model claude-3-5-sonnet-20241022 --modes l,v,vl
  
  # Debug with limited samples
  python 01_inference_claude.py --model claude-3-5-haiku-20241022 --modes l --debug-samples 5
  
  # Specific tasks with custom worker count
  python 01_inference_claude.py --model claude-3-5-sonnet-20241022 --tasks fork,legal --modes vl --max-workers 5
  
  # Sequential processing (no parallel)
  python 01_inference_claude.py --model claude-3-5-haiku-20241022 --modes l --no-parallel
        """
    )
    
    # Model configuration
    parser.add_argument('--model', required=True, 
                       help='Claude model name (e.g., claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022)')
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
    
    args = parser.parse_args()
    
    try:
        # Parse modes and tasks
        modes = parse_mode_list(args.modes)
        tasks = parse_task_list(args.tasks)
        
        # Handle parallel processing arguments
        parallel = args.parallel and not args.no_parallel
        
        # Initialize runner
        runner = ClaudeInferenceRunner(
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
        
        print(f"\n🎉 Inference Pipeline Completed!")
        print(f"   Model: {args.model}")
        print(f"   Processed: {total_processed} samples")
        print(f"   Next step: python 02_extract.py --model {runner.normalized_name}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()