import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from PIL import Image
from tqdm import tqdm

# Suppress noisy vLLM tokenizer warnings
warnings.filterwarnings("ignore", message="The following intended overrides are not keyword args and will be dropped")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.config import *
from config.config import RESULTS_DIR, BENCHMARK_ROOT
from utils.task_loader import TaskLoader
from utils.utils import get_model_safe_name


def normalize_model_name(model_name: str) -> str:
    """Convert model names to filesystem-safe format"""
    return model_name.replace('/', '-').replace('_', '-').lower()

def setup_huggingface_auth():
    """Setup HuggingFace authentication for gated models"""
    try:
        # Try loading from JSON config first
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'api_keys.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                hf_token = config.get('huggingface', {}).get('api_key')
                if hf_token:
                    os.environ['HF_TOKEN'] = hf_token
                    print(f"   🔑 HuggingFace authentication configured")
                    return True
    except Exception as e:
        print(f"   ⚠️  Could not load HF token from config: {e}")
    
    # Fallback to environment variable
    if os.getenv('HF_TOKEN'):
        print(f"   🔑 Using HuggingFace token from environment")
        return True
    
    print(f"   ⚠️  No HuggingFace token found - gated models may fail")
    return False

class VLLMInferenceRunner:
    """Pure vLLM inference runner - no answer extraction"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.normalized_name = normalize_model_name(model_name)
        self.max_model_len = MAX_MODEL_LENGTH
        self.gpu_memory_utilization = GPU_MEMORY_UTILIZATION
        self.tensor_parallel_size = TENSOR_PARALLEL_SIZE
        self.raw_prompt_warnings_shown = set()  # Track which modes we've warned about
        
        print(f"🚀 Initializing vLLM Inference Runner")
        print(f"   Model: {model_name}")
        print(f"   Normalized: {self.normalized_name}")
        print(f"   Max model length: {self.max_model_len}")
        print(f"   GPU memory utilization: {self.gpu_memory_utilization}")
        print(f"   Tensor parallel size: {self.tensor_parallel_size} GPUs")
        
        # Setup HuggingFace authentication for gated models
        setup_huggingface_auth()
        
        # Setup results directory
        self.results_dir = RESULTS_DIR / self.normalized_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.results_dir / "output.jsonl"
        
        # Initialize vLLM model
        try:
            self.llm = LLM(
                model=model_name,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=self.tensor_parallel_size,
                max_num_seqs=MAX_NUM_SEQS,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 1}  # Allow 1 image per prompt
            )
            
            # Configure sampling parameters - use model defaults except max_tokens
            self.sampling_params = SamplingParams(max_tokens=MODEL_MAX_TOKENS)
            
            # Initialize tokenizer for chat template formatting
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    trust_remote_code=True
                )
                self.has_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None
                print(f"   Chat template: {'✅ available' if self.has_chat_template else '❌ not available'}")
            except Exception as e:
                print(f"   Warning: Could not load tokenizer: {e}")
                self.tokenizer = None
                self.has_chat_template = False
            
            print(f"✅ Model loaded successfully!")
            print(f"   Sampling: max_tokens={self.sampling_params.max_tokens}, temperature=model_default")
            
        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            raise
    
    def run_inference(self, benchmark_root: str, modes: List[str], tasks: List[str] = None,
                     debug_samples: int = 0, resume: bool = True) -> int:
        """Run inference across specified modes and tasks"""
        
        task_loader = TaskLoader(benchmark_root)
        all_tasks = tasks if tasks else task_loader.all_tasks
        
        print(f"\n📊 Starting SEAM Inference")
        print(f"   Tasks: {len(all_tasks)} ({', '.join(all_tasks)})")
        print(f"   Modes: {', '.join(modes)}")
        print(f"   Debug samples: {debug_samples if debug_samples > 0 else 'All'}")
        print(f"   Output: {self.output_file}")
        print(f"   Resume: {resume}")
        
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
        
        # Generate all prompts first
        all_prompts = []
        for mode in modes:
            prompts = self.generate_prompts(task_loader, all_tasks, mode, debug_samples, existing_results)
            all_prompts.extend(prompts)
        
        print(f"📝 Total prompts generated: {len(all_prompts)}")
        
        if not all_prompts:
            print("✅ No new prompts to process!")
            return 0
        
        print(f"\n🧠 Running batch inference on {len(all_prompts)} prompts...")
        
        # Run inference in batches
        batch_size = 1000  # Process in chunks to avoid memory issues
        all_results = []
        
        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i:i + batch_size]
            print(f"🔄 Processing batch {i//batch_size + 1}/{(len(all_prompts)-1)//batch_size + 1} ({len(batch_prompts)} samples)")
            
            batch_results = self.process_batch(batch_prompts)
            all_results.extend(batch_results)
            
            # Save incrementally
            self.save_batch_results(batch_results)
            total_processed += len(batch_results)
            
            print(f"✅ Completed batch {i//batch_size + 1}: {len(batch_results)} samples")
        
        total_time = time.time() - start_time
        throughput = total_processed / total_time if total_time > 0 else 0
        
        print(f"\n✅ Inference Completed!")
        print(f"   Total processed: {total_processed}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Throughput: {throughput:.1f} samples/sec")
        print(f"   Results saved: {self.output_file}")
        
        return total_processed
    
    def generate_prompts(self, task_loader: TaskLoader, tasks: List[str], 
                        mode: str, debug_samples: int, existing_results: set) -> List[dict]:
        """Generate prompts for specified tasks and mode"""
        prompts = []
        
        progress_bar = tqdm(tasks, desc=f"Generating {mode} prompts")
        
        for task_name in (progress_bar if progress_bar else tasks):
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
                        }
                        
                        prompts.append(prompt_data)
                        
                    except Exception as e:
                        print(f"⚠️  Error formatting {task_name}[{idx}]: {e}")
                        continue
                        
            except Exception as e:
                print(f"⚠️  Error loading task {task_name}: {e}")
                continue
        
        if progress_bar:
            progress_bar.close()
            
        print(f"📝 Generated {len(prompts)} new prompts for mode '{mode}'")
        return prompts
    
    def format_prompt_for_model(self, prompt_text: str, mode: str, image_path: str = None) -> str:
        """Format prompt according to model's capabilities and mode"""
        if not self.has_chat_template or not self.tokenizer:
            # Fall back to raw prompt for models without chat templates
            if mode not in self.raw_prompt_warnings_shown:
                print(f"⚠️  Using raw prompt text for {mode.upper()} mode - no chat template available")
                self.raw_prompt_warnings_shown.add(mode)
            return prompt_text
        
        try:
            if mode == "l":
                # Language-only mode
                messages = [
                    {"role": "system", "content": MODEL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text}
                ]
                
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
            elif mode in ["v", "vl"]:
                # Vision modes - use model-specific chat template formatting
                if "internvl" in self.model_name.lower():
                    # InternVL models expect string content with image placeholders
                    messages = [
                        {"role": "system", "content": MODEL_SYSTEM_PROMPT},
                        {"role": "user", "content": f"<image>\n{prompt_text}"}
                    ]
                else:
                    # Other models (Qwen, LLaVA, etc.) use list content format
                    messages = [
                        {"role": "system", "content": MODEL_SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text}
                        ]}
                    ]
                
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            
        except Exception as e:
            print(f"⚠️  Chat template formatting failed: {e}")
            # Fall back to raw prompt
            return prompt_text
        
        return prompt_text
    
    def process_batch(self, prompts: List[dict]) -> List[dict]:
        """Process a batch of prompts through vLLM with proper multimodal formatting"""
        from utils.task_loader import TaskLoader
        task_loader = TaskLoader(str(BENCHMARK_ROOT))
        
        # Pre-load all task data to avoid repeated loading
        task_data_cache = {}
        for prompt_data in prompts:
            task_name = prompt_data["task_name"]
            if task_name not in task_data_cache:
                task_data_cache[task_name] = task_loader.load_task_data(task_name)
        
        # Prepare vLLM inputs in the correct format
        vllm_inputs = []
        
        for prompt_data in prompts:
            mode = prompt_data["mode"]
            task_name = prompt_data["task_name"]
            
            if mode == "l":
                # Language-only mode - simple string format
                formatted_prompt = self.format_prompt_for_model(
                    prompt_data["question"], 
                    mode
                )
                vllm_inputs.append(formatted_prompt)
                
            elif mode in ["v", "vl"]:
                # Vision modes - use vLLM multimodal format
                try:
                    # Get the original sample from cache
                    task_data = task_data_cache[task_name]
                    sample = task_data[prompt_data["index"]]
                    image_path = task_loader.get_image_path(task_name, sample)
                    
                    if mode == "v":
                        # Vision-only: get vision-specific prompt
                        vision_prompt = task_loader.get_vision_only_prompt(task_name, sample)
                        formatted_prompt = self.format_prompt_for_model(vision_prompt, mode)
                    else:
                        # Vision-language: use full prompt
                        formatted_prompt = self.format_prompt_for_model(prompt_data["question"], mode)
                    
                    # Load image with PIL
                    if image_path and os.path.exists(image_path):
                        image = Image.open(image_path)
                        
                        # Use vLLM's multimodal format
                        vllm_input = {
                            "prompt": formatted_prompt,
                            "multi_modal_data": {"image": image}
                        }
                        vllm_inputs.append(vllm_input)
                    else:
                        print(f"⚠️  Image not found for {task_name}[{prompt_data['index']}]: {image_path}")
                        # Fallback to text-only
                        formatted_prompt = self.format_prompt_for_model(prompt_data["question"], "l")
                        vllm_inputs.append(formatted_prompt)
                        
                except Exception as e:
                    print(f"⚠️  Error processing vision input for {task_name}[{prompt_data['index']}]: {e}")
                    # Fallback to text-only
                    formatted_prompt = self.format_prompt_for_model(prompt_data["question"], "l")
                    vllm_inputs.append(formatted_prompt)
        
        # Run batch inference
        try:
            outputs = self.llm.generate(vllm_inputs, self.sampling_params)
                
        except Exception as e:
            print(f"⚠️  Batch generation failed: {e}")
            raise
        
        # Combine results with metadata
        results = []
        for prompt_data, output in zip(prompts, outputs):
            result = {
                **prompt_data,  # Include all original fields
                "output": output.outputs[0].text,
            }
            results.append(result)
        
        return results
    
    def save_batch_results(self, results: List[dict]):
        """Append batch results to output file"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
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
        description='SEAM Benchmark vLLM Inference Only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference for all tasks and modes
  python 01_inference_vllm.py --model Qwen/Qwen2.5-VL-7B-Instruct --modes l,v,vl
  
  # Debug with limited samples
  python 01_inference_vllm.py --model Qwen/Qwen2.5-VL-7B-Instruct --modes l --debug-samples 10
  
  # Specific tasks and custom GPU settings
  python 01_inference_vllm.py --model Qwen/Qwen2.5-VL-7B-Instruct --tasks fork,legal --gpu-memory-utilization 0.6
        """
    )
    
    # Model configuration
    parser.add_argument('--model', required=True, 
                       help='Model name (e.g., Qwen/Qwen2.5-VL-7B-Instruct)')
    
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
    
    args = parser.parse_args()
    
    try:
        # Parse modes and tasks
        modes = parse_mode_list(args.modes)
        tasks = parse_task_list(args.tasks)
        
        # Initialize runner
        runner = VLLMInferenceRunner(model_name=args.model)
        
        # Run inference
        total_processed = runner.run_inference(
            benchmark_root=str(BENCHMARK_ROOT),
            modes=modes,
            tasks=tasks,
            debug_samples=args.debug_samples,
            resume=not args.no_resume
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