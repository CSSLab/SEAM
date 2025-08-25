#!/usr/bin/env python3
"""
SEAM Benchmark - Unified Answer Extraction
Extracts final answers from model outputs using local extraction model.
Modifies output.jsonl in-place to create extracted.jsonl with final answers.
"""

import argparse
import json
import os
import sys
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Third-party imports
from vllm import LLM, SamplingParams

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Local imports
from config.config import (
    EXTRACTION_MAX_TOKENS, EXTRACTION_TEMPERATURE, EXTRACTION_MODEL, EXTRACTION_SYSTEM_PROMPT,
    RESULTS_DIR, GPU_MEMORY_UTILIZATION, TENSOR_PARALLEL_SIZE, MAX_MODEL_LENGTH
)
from tqdm import tqdm

class UnifiedAnswerExtractor:
    """Unified answer extractor using vLLM for all model types"""
    
    def __init__(self, extraction_model: str = None, gpu_memory_utilization: float = None, 
                 tensor_parallel_size: int = None, max_model_len: int = None):
        self.extraction_model = extraction_model or EXTRACTION_MODEL
        print(f"🔧 Loading extraction model: {self.extraction_model}")
        
        # Use command-line overrides or defaults
        default_tps = tensor_parallel_size if tensor_parallel_size is not None else TENSOR_PARALLEL_SIZE
        default_gpu_util = gpu_memory_utilization if gpu_memory_utilization is not None else GPU_MEMORY_UTILIZATION
        default_max_len = max_model_len if max_model_len is not None else MAX_MODEL_LENGTH
        
        # Try different vLLM configurations if the initial one fails
        fallback_configs = [
            # Original config (current settings or CLI overrides)
            {
                'tensor_parallel_size': default_tps,
                'gpu_memory_utilization': default_gpu_util,
                'max_model_len': default_max_len
            },
            # Single GPU with reduced memory for extraction
            {
                'tensor_parallel_size': 1,
                'gpu_memory_utilization': 0.4,
                'max_model_len': 8192
            },
            # Very conservative settings
            {
                'tensor_parallel_size': 1,
                'gpu_memory_utilization': 0.3,
                'max_model_len': 4096
            },
            # Minimal settings for debugging
            {
                'tensor_parallel_size': 1,
                'gpu_memory_utilization': 0.2,
                'max_model_len': 2048
            }
        ]
        
        self.llm = None
        for i, config in enumerate(fallback_configs):
            config_desc = f"configuration {i+1} ({', '.join([f'{k}={v}' for k, v in config.items()])})"
            print(f"  Attempting to load extraction model with {config_desc}")
            
            try:
                # Initialize vLLM model for batch extraction
                self.llm = LLM(
                    model=self.extraction_model,
                    tensor_parallel_size=config['tensor_parallel_size'],
                    gpu_memory_utilization=config['gpu_memory_utilization'],
                    max_model_len=config['max_model_len'],
                    trust_remote_code=True
                )
                
                # Set up sampling parameters for extraction
                self.sampling_params = SamplingParams(
                    temperature=EXTRACTION_TEMPERATURE,
                    max_tokens=EXTRACTION_MAX_TOKENS,
                    stop=None
                )
                
                print(f"✅ Extraction model loaded successfully with {config_desc}!")
                break
                
            except Exception as e:
                error_str = str(e).lower()
                if "memory" in error_str or "cuda" in error_str or "vllm" in error_str or "workerproc" in error_str:
                    print(f"  ⚠️  GPU/Memory error with {config_desc}: {str(e)[:200]}...")
                    if i < len(fallback_configs) - 1:
                        print(f"  Trying next configuration...")
                    continue
                else:
                    # Non-memory error, don't try other configs
                    print(f"❌ Failed to load extraction model: {e}")
                    raise
        
        if self.llm is None:
            raise RuntimeError("Failed to load extraction model with any configuration")
    
    def extract_answer_regex_only(self, output: str) -> tuple:
        """
        Extract final answer using only regex
        Returns: (final_answer, extraction_method) or None if failed
        """
        answer = self.regex_extract(output)
        if answer in ["A", "B", "C", "D"]:
            return answer, "regex"
        return None
    
    def batch_llm_extract(self, outputs: List[str]) -> List[str]:
        """
        Batch extract answers using vLLM for multiple outputs
        Returns list of extracted answers (A/B/C/D/Z)
        """
        if not outputs:
            return []
        
        # Create extraction prompts for all outputs
        prompts = []
        for output in outputs:
            extraction_prompt = (
                f"Here is the complete predicted answer for a multiple-choice question\n\n"
                f"***{output}***\n\n"
                f"Your task: Extract the final answer (the best option) from the text above.\n"
                f"Ignore the reasoning process and any inconsistence in the above complete predicted answer.\n"
                f"It is usually in the format 'The best option is [letter]' at the end of the complete predicted answer.\n"
                f"Reply with ONLY ONE LETTER: A, B, C, D, or Z.\n"
                f"Do not include any other text, just the single letter."
            )
            
            # Create conversation format
            conversation = [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": extraction_prompt}
            ]
            
            # Convert to prompt format
            try:
                prompt = self.llm.get_tokenizer().apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # Fallback for non-chat models
                prompt = f"System: {EXTRACTION_SYSTEM_PROMPT}\n\nUser: {extraction_prompt}\n\nAssistant:"
            
            prompts.append(prompt)
        
        try:
            # Generate using vLLM batch processing
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            # Extract responses
            results = []
            for output in outputs:
                response = output.outputs[0].text.strip().upper()
                # Only accept exact matches to A, B, C, D
                if response in ["A", "B", "C", "D"]:
                    results.append(response)
                else:
                    results.append("Z")
            
            return results
            
        except Exception as e:
            print(f"⚠️  Batch LLM extraction failed: {e}")
            return ["Z"] * len(prompts)
    
    def llm_extract(self, output: str) -> str:
        """Extract final answer using vLLM"""
        extraction_prompt = (
            f"Here is the complete predicted answer for a multiple-choice question\n\n"
            f"***{output}***\n\n"
            f"Your task: Extract the final answer (the best option) from the text above.\n"
            f"Ignore the reasoning process and any inconsistence in the above complete predicted answer.\n"
            f"It is usually in the format 'The best option is [letter]' at the end of the complete predicted answer.\n"
            f"Reply with ONLY ONE LETTER: A, B, C, D, or Z.\n"
            f"Do not include any other text, just the single letter."
        )
        
        try:
            # Create conversation format
            conversation = [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": extraction_prompt}
            ]
            
            # Convert to prompt format - check if model supports chat template
            try:
                # Try to apply chat template (works for chat models)
                prompt = self.llm.get_tokenizer().apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # Fallback for non-chat models
                prompt = f"System: {EXTRACTION_SYSTEM_PROMPT}\n\nUser: {extraction_prompt}\n\nAssistant:"
            
            # Generate using vLLM
            outputs = self.llm.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            # Clean the extracted answer
            extracted = response.strip().upper()
            
            # Only accept EXACT matches to A, B, C, D, or Z
            if extracted in ["A", "B", "C", "D", "Z"]:
                return extracted
            else:
                return "Z"
                
        except Exception as e:
            print(f"⚠️  LLM extraction failed: {e}")
            return "Z"
    
    def regex_extract(self, output: str) -> str:
        """Very strict regex - only matches 'The best option is [letter]' format"""
        # Focus on the last 30 characters to find the exact pattern
        tail = output[-30:] if len(output) > 30 else output
        
        # Only match the exact format provided in the prompt
        try:
            match = re.search(r"The best option is ([A-D])", tail, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        except re.error:
            pass
        
        return "Z"

def find_all_models() -> List[str]:
    """Find all model directories in results/"""
    results_dir = RESULTS_DIR
    if not results_dir.exists():
        return []
    
    models = []
    for item in results_dir.iterdir():
        if item.is_dir() and (item / "output.jsonl").exists():
            models.append(item.name)
    
    return sorted(models)

def process_model(model_name: str, extraction_model: str = None, 
                  gpu_memory_utilization: float = None, tensor_parallel_size: int = None,
                  max_model_len: int = None) -> Dict[str, int]:
    """Process extraction for a single model"""
    results_dir = RESULTS_DIR / model_name
    output_file = results_dir / "output.jsonl"
    extracted_file = results_dir / "extracted.jsonl"
    
    if not output_file.exists():
        print(f"❌ Output file not found: {output_file}")
        return {"processed": 0, "errors": 0}
    
    print(f"\\n📄 Processing model: {model_name}")
    print(f"   Input: {output_file}")
    print(f"   Output: {extracted_file}")
    
    # Always run extraction (removed check for existing file)
    
    # Load all results
    results = []
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        result = json.loads(line.strip())
                        results.append(result)
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️  JSON decode error on line {line_num}: {e}")
                        continue
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return {"processed": 0, "errors": 1}
    
    if not results:
        print(f"   No valid results found in {output_file}")
        return {"processed": 0, "errors": 1}
    
    print(f"   Loaded {len(results)} results")
    
    # Check which results need extraction
    pending_results = [r for r in results if "final_answer" not in r or r.get("final_answer") == "PENDING"]
    
    if not pending_results:
        print(f"   All results already have extracted answers!")
        # Copy to extracted file if it doesn't exist
        if not extracted_file.exists():
            shutil.copy2(output_file, extracted_file)
            print(f"   ✅ Copied to {extracted_file}")
        return {"processed": len(results), "errors": 0}
    
    print(f"   Found {len(pending_results)} results needing extraction")
    
    # Initialize extractor
    extractor = UnifiedAnswerExtractor(extraction_model, gpu_memory_utilization, 
                                     tensor_parallel_size, max_model_len)
    
    # Two-pass extraction process
    processed_count = 0
    error_count = 0
    regex_count = 0
    llm_count = 0
    failed_count = 0
    
    print("   🔍 Pass 1: Regex extraction...")
    
    # Pass 1: Try regex on all results
    llm_needed = []  # Store results that need LLM extraction
    llm_indices = []  # Store their indices in the results array
    
    for i, result in enumerate(results):
        try:
            # Skip if already extracted
            if "final_answer" in result and result.get("final_answer") != "PENDING":
                continue
            
            # Try regex extraction
            regex_result = extractor.extract_answer_regex_only(result["output"])
            
            if regex_result is not None:
                # Regex succeeded
                final_answer, method = regex_result
                result["final_answer"] = final_answer
                result["correct"] = final_answer == result["answer"]
                result["extraction_method"] = method
                regex_count += 1
                processed_count += 1
            else:
                # Regex failed, queue for LLM
                llm_needed.append(result["output"])
                llm_indices.append(i)
            
        except Exception as e:
            print(f"   ⚠️  Error in regex extraction for {result.get('task_name', 'unknown')}[{result.get('index', 'unknown')}]: {e}")
            error_count += 1
            continue
    
    print(f"   ✅ Regex extracted: {regex_count}")
    print(f"   ⏳ Need LLM extraction: {len(llm_needed)}")
    
    # Pass 2: Batch LLM extraction for failed cases
    if llm_needed:
        print("   🤖 Pass 2: Batch LLM extraction...")
        try:
            llm_results = extractor.batch_llm_extract(llm_needed)
            
            # Update results with LLM extractions
            for idx, llm_result in zip(llm_indices, llm_results):
                result = results[idx]
                result["final_answer"] = llm_result
                result["correct"] = llm_result == result["answer"]
                result["extraction_method"] = "llm" if llm_result in ["A", "B", "C", "D"] else "failed"
                
                if llm_result in ["A", "B", "C", "D"]:
                    llm_count += 1
                else:
                    failed_count += 1
                processed_count += 1
                
        except Exception as e:
            print(f"   ⚠️  Batch LLM extraction failed: {e}")
            # Mark all LLM-needed results as failed
            for idx in llm_indices:
                result = results[idx]
                result["final_answer"] = "Z"
                result["correct"] = False
                result["extraction_method"] = "failed"
                failed_count += 1
                processed_count += 1
                error_count += 1
    
    # Save extracted results
    try:
        with open(extracted_file, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"   ✅ Saved {len(results)} results to {extracted_file}")
        
    except Exception as e:
        print(f"   ❌ Error saving extracted results: {e}")
        error_count += 1
    
    # Compute and display statistics
    total_correct = sum(1 for r in results if r.get("correct", False))
    total_samples = len(results)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    print(f"   📊 Extraction Results:")
    print(f"      Total samples: {total_samples}")
    print(f"      Newly processed: {processed_count}")
    print(f"      Correct answers: {total_correct}")
    print(f"      Overall accuracy: {accuracy:.3f} ({100*accuracy:.1f}%)")
    
    # Show extraction method breakdown for newly processed samples
    if processed_count > 0:
        print(f"   🔧 Extraction Method Breakdown (newly processed):")
        print(f"      Regex extractions: {regex_count}")
        print(f"      LLM extractions: {llm_count}")
        print(f"      Failed extractions: {failed_count}")
    
    # Show overall extraction method breakdown
    method_counts = {}
    for result in results:
        method = result.get("extraction_method", "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1
    
    if method_counts:
        print(f"   📈 Overall Extraction Methods: {dict(method_counts)}")
    
    return {"processed": processed_count, "errors": error_count}

def main():
    parser = argparse.ArgumentParser(
        description='SEAM Benchmark - Unified Answer Extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract for specific model
  python 02_extract.py --model qwen-qwen2.5-vl-7b-instruct
  
  # Extract for all models
  python 02_extract.py --all
  
  # Use custom extraction model
  python 02_extract.py --model gpt-4o-mini --extraction-model Qwen/Qwen2.5-1.5B-Instruct
  
  # Override GPU settings
  python 02_extract.py --model gpt-4o-mini --gpu-memory-utilization 0.4 --tensor-parallel-size 1
        """
    )
    
    # Model selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', help='Specific model name to extract')
    group.add_argument('--all', action='store_true', help='Extract for all models')
    group.add_argument('--list', action='store_true', help='List available models')
    
    # Extraction configuration
    parser.add_argument('--extraction-model', default=None,
                       help=f'Extraction model (default: {EXTRACTION_MODEL})')
    # Force parameter removed - always rerun extraction
    
    # GPU configuration overrides
    parser.add_argument('--gpu-memory-utilization', type=float, default=None,
                       help='Override GPU memory utilization (0.0-1.0)')
    parser.add_argument('--tensor-parallel-size', type=int, default=None,
                       help='Override tensor parallel size')
    parser.add_argument('--max-model-len', type=int, default=None,
                       help='Override max model length')
    
    args = parser.parse_args()
    
    try:
        if args.list:
            # List available models
            models = find_all_models()
            if not models:
                print("No models found in results/")
                return 0
            
            print(f"Available models ({len(models)}):")
            for model in models:
                results_dir = RESULTS_DIR / model
                output_file = results_dir / "output.jsonl"
                extracted_file = results_dir / "extracted.jsonl"
                
                status = "✅ extracted" if extracted_file.exists() else "⏳ pending"
                
                # Count samples
                sample_count = 0
                if output_file.exists():
                    try:
                        with open(output_file, 'r') as f:
                            sample_count = sum(1 for line in f if line.strip())
                    except:
                        pass
                
                print(f"  • {model} ({sample_count} samples) - {status}")
            
            return 0
        
        elif args.all:
            # Process all models
            models = find_all_models()
            if not models:
                print("No models found in results/")
                return 1
            
            print(f"🔍 Processing {len(models)} models...")
            
            total_stats = {"processed": 0, "errors": 0}
            for model in models:
                stats = process_model(model, args.extraction_model, args.gpu_memory_utilization,
                                    args.tensor_parallel_size, args.max_model_len)
                total_stats["processed"] += stats["processed"]
                total_stats["errors"] += stats["errors"]
            
            print(f"\\n🎉 Batch Extraction Completed!")
            print(f"   Total processed: {total_stats['processed']}")
            print(f"   Total errors: {total_stats['errors']}")
            
        else:
            # Process specific model
            stats = process_model(args.model, args.extraction_model, args.gpu_memory_utilization,
                                args.tensor_parallel_size, args.max_model_len)
            
            print(f"\\n🎉 Extraction Completed!")
            print(f"   Processed: {stats['processed']}")
            print(f"   Errors: {stats['errors']}")
            
            if stats["processed"] > 0:
                print(f"   Next step: python 03_metric.py --model {args.model}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)