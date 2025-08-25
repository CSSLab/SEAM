# SEAM: Semantically Equivalent Across Modalities Benchmark for Vision-Language Models

## Abstract

Evaluating whether vision–language models (VLMs) reason consistently across representations is challenging because modality comparisons are typically confounded by task differences and asymmetric information. We introduce **SEAM**, a benchmark that pairs semantically equivalent inputs across four domains with existing standardized textual and visual notations. By employing distinct notation systems across modalities, in contrast to OCR-based image-text pairing, SEAM provides a rigorous comparative assessment of the textual-symbolic and visual-spatial reasoning capabilities of VLMs. Across 16 contemporary models, we observe systematic modality imbalance: vision frequently lags language in overall performance, despite the problems containing semantically equivalent information, and cross-modal agreement is relatively low. Our error analysis reveals two main drivers: textual perception failures from tokenization in domain notations and visual perception failures that induce hallucinations. We also show that our results are largely robust to visual transformations. SEAM establishes a controlled, semantically equivalent setting for measuring and improving modality-agnostic reasoning.

## Features

- 3,200 base questions across 4 domains (Chess, Chemistry, Music, Graph Theory) = 9,600 total evaluations across 3 modalities
- 3 modalities: Language-only, Vision-only, Vision-Language
- Unified 3-stage pipeline: Inference → Extraction → Metrics
- Support for vLLM, OpenAI, and Claude models

## Setup

1. **Install dependencies**:
   ```bash
   cd code/config
   pip install -r requirements.txt
   ```

2. **Configure API keys**:
   ```bash
   cp api_keys.json.template api_keys.json
   ```
   Edit `api_keys.json` with your API keys:
   ```json
   {
     "openai": {
       "api_key": "sk-your-openai-key-here"
     },
     "anthropic": {
       "api_key": "your-anthropic-key-here"  
     },
     "huggingface": {
       "api_key": "your-huggingface-token-here"
     }
   }
   ```

## Quick Start

Run a complete evaluation in 4 steps:

**1. Dataset** - Automatically downloaded on first run:
```bash
# No manual setup required - dataset downloads automatically when needed

# Or manually download using HuggingFace datasets:
python -c "
from datasets import load_dataset
dataset = load_dataset('lilvjosephtang/SEAM-Benchmark')
print('Dataset downloaded successfully')
print(f'Available tasks: {list(dataset.keys())}')
"
```

**2. Inference** - Generate model responses:
```bash
# Choose your model provider:
cd code/run && python 01_inference_vllm.py --model Qwen/Qwen2.5-VL-7B-Instruct --modes l,v,vl
cd code/run && python 01_inference_openai.py --model gpt-4o-mini --modes l,v,vl  
cd code/run && python 01_inference_claude.py --model claude-3-5-sonnet-20241022 --modes l,v,vl
```

**3. Extract answers** - Parse model outputs:
```bash
cd code/run && python 02_extract.py --model qwen-qwen2.5-vl-7b-instruct
```

**4. Compute metrics** - Analyze results:
```bash
cd code/run && python 03_metric.py --model qwen-qwen2.5-vl-7b-instruct
```

### Multi-Provider Model Support

The benchmark supports running multiple model providers with automatic script detection:

```bash
# Run complete pipeline for all enabled models
./run_all_models.sh

# The script automatically detects model type and uses appropriate inference script:
# - vLLM models -> 01_inference_vllm.py
# - OpenAI models -> 01_inference_openai.py  
# - Claude models -> 01_inference_claude.py
```

Edit `run_all_models.sh` to enable/disable specific models by commenting/uncommenting them:

```bash
VLLM_MODELS=(
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "OpenGVLab/InternVL3-8B" 
    # "meta-llama/Llama-3.2-11B-Vision-Instruct"  # Commented out
)

OPENAI_MODELS=(
    # "gpt-4o-mini"  # Add your OpenAI models here
)

CLAUDE_MODELS=(
    # "claude-3-5-sonnet-20241022"  # Add your Claude models here
)
```

### Advanced Usage

#### Advanced vLLM Options
```bash
# Specific tasks only
cd code/run && python 01_inference_vllm.py --model Qwen/Qwen2.5-VL-7B-Instruct \
                                           --tasks fork,legal,puzzle

# Debug mode with limited samples
cd code/run && python 01_inference_vllm.py --model Qwen/Qwen2.5-VL-7B-Instruct \
                                           --debug-samples 10 --modes l
```

#### OpenAI Batch Processing Step-by-Step
```bash
# Prepare batch data
cd code/run && python 01_inference_openai.py --model gpt-4o-mini --batch --action prepare --modes l,v,vl

# Submit batch job
cd code/run && python 01_inference_openai.py --model gpt-4o-mini --batch --action submit

# Check status
cd code/run && python 01_inference_openai.py --model gpt-4o-mini --batch --action status

# Download results when complete
cd code/run && python 01_inference_openai.py --model gpt-4o-mini --batch --action download
```

#### Advanced Claude Options
```bash
# Custom parallel processing settings
cd code/run && python 01_inference_claude.py --model claude-3-5-sonnet-20241022 \
                                             --modes l,v,vl \
                                             --max-workers 5

# Sequential processing (no parallel)
cd code/run && python 01_inference_claude.py --model claude-3-5-haiku-20241022 \
                                             --modes l \
                                             --no-parallel

# Debug with limited samples
cd code/run && python 01_inference_claude.py --model claude-3-5-sonnet-20241022 \
                                             --debug-samples 10 --modes l
```

#### Resume and Error Recovery
```bash
# Resume interrupted inference (automatic)
cd code/run && python 01_inference_vllm.py --model Qwen/Qwen2.5-VL-7B-Instruct --modes l,v,vl

# Force re-extraction
cd code/run && python 02_extract.py --model qwen-qwen2.5-vl-7b-instruct --force

# List available models
cd code/run && python 02_extract.py --list
cd code/run && python 03_metric.py --list
```

## Directory Structure

```
seam-benchmark/
├── code/                     # All executable code
│   ├── run/                  # Main execution scripts (unified pipeline)
│   │   ├── 01_inference_vllm.py    # vLLM inference (local models)
│   │   ├── 01_inference_openai.py  # OpenAI inference (real-time + batch)
│   │   ├── 01_inference_claude.py  # Claude inference (real-time)
│   │   ├── 02_extract.py          # Unified answer extraction
│   │   ├── 03_metric.py           # Unified metrics & comparison plots
│   │   └── legacy/               # Deprecated scripts (not in git)
│   ├── config/              # Configuration and setup
│   │   ├── config.py        # Unified benchmark configuration
│   │   ├── requirements.txt # Python dependencies
│   │   ├── setup.sh         # Environment setup script
│   │   ├── api_keys.json    # API keys configuration (not in git)
│   │   └── api_keys.json.template # API keys template
│   ├── dataset/             # Dataset generation scripts
│   │   ├── dataset_*.py     # Domain-specific dataset generators
│   │   ├── hf_dataset.py    # HuggingFace dataset upload/management
│   │   ├── tasks.py         # Task definitions
│   │   └── graph_tasks/     # Graph theory task implementations
│   └── utils/               # Utility functions
│       ├── task_loader.py   # Unified task loading (HF + JSONL support)
│       └── utils.py         # Image processing utilities
├── data/                    # HuggingFace dataset cache (parquet files)
│   ├── {task}.parquet       # e.g., fork.parquet, carbon.parquet
│   └── ...                  # Downloaded automatically from HuggingFace
├── raw_data/               # Original JSONL datasets + images (fallback)
│   ├── {task}.jsonl        # e.g., fork.jsonl, carbon.jsonl
│   ├── {task}/             # Task-specific images
│   │   └── *.png           # Individual task images
│   └── ...                 # Generated by dataset scripts or manual download
├── results/                 # Model evaluation results (unified structure)
│   ├── {model-name}/        # e.g., qwen-qwen2.5-vl-7b-instruct/
│   │   ├── output.jsonl     # Raw inference outputs
│   │   ├── extracted.jsonl  # With final answers extracted
│   │   ├── metrics.json     # Computed metrics
│   │   ├── batch/           # Batch processing files (OpenAI)
│   │   └── plots/           # Generated visualizations
│   └── comparison_plots.png # Multi-model comparison
└── logs/                    # Application logs
```

## Configuration

### Data Source Configuration

The benchmark supports two data sources with seamless switching:

```python
# In code/config/config.py
USE_HF_DATASET = True  # Use HuggingFace dataset (recommended)
HF_REPO_ID = "lilvjosephtang/SEAM-Benchmark"  # HuggingFace repository
BENCHMARK_ROOT = Path("/path/to/raw_data")  # Fallback for JSONL files
HF_DATA_ROOT = Path("/path/to/hf_data")  # HuggingFace parquet cache
```

**Automatic Data Source Selection:**
- When `USE_HF_DATASET = True`: Downloads and uses HuggingFace parquet files
- When `USE_HF_DATASET = False`: Uses local JSONL files from `BENCHMARK_ROOT`
- **100% Identical Results**: Both data sources produce exactly the same evaluation prompts and results
- **Automatic Fallback**: If HF dataset fails to load, automatically falls back to JSONL files

### Provider-Specific Settings

#### vLLM (Local Models)
- **Sampling**: Uses model defaults except max_tokens limit
- **Resume**: Automatic detection and skipping of completed samples
- **Memory**: Configurable GPU memory utilization and model length
- **Model Compatibility**: Automatic chat template detection for InternVL and other model families
- **Warning Suppression**: Filtered noisy tokenizer warnings for cleaner logs

#### OpenAI API
- **Real-time**: Direct API calls with rate limiting
- **Batch**: Full batch API support (prepare→submit→monitor→download)
- **Vision**: Base64 image encoding for multimodal tasks

#### Claude API
- **Real-time**: Direct API calls with Anthropic client and exponential backoff retry
- **Parallel Processing**: ThreadPoolExecutor with configurable worker count (default: 10)
- **Vision**: Proper image format handling and content structure
- **Resilient**: Multi-round retry system ensures all prompts are processed

### Environment Variables

```bash
# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1    # Specify which GPUs to use
export VLLM_WORKER_MULTIPROC_METHOD=spawn  # For multi-GPU setups

# Model caching
export HF_HOME=/path/to/cache       # HuggingFace model cache directory
```

### Command Line Options

```bash
# 01_inference_vllm.py options
--model MODEL              # Model name (e.g., Qwen/Qwen2.5-VL-7B-Instruct)
--modes l,v,vl            # Comma-separated modes (default: l)
--tasks task1,task2       # Comma-separated task names (default: all)
--debug-samples N        # Limit samples for debugging
--no-resume             # Disable resume functionality

# 01_inference_openai.py options
--model MODEL            # OpenAI model (e.g., gpt-4o-mini)
--batch                  # Use batch API instead of real-time
--action ACTION          # Batch action: prepare|submit|status|download|all
--temperature TEMP       # Temperature (default: 0.7)
--max-tokens N          # Max tokens (default: 8192)

# 01_inference_claude.py options
--model MODEL           # Claude model (e.g., claude-3-5-sonnet-20241022)
--max-tokens N         # Max tokens (default: 8192)
--no-parallel         # Disable parallel processing (use sequential)
--max-workers N       # Maximum parallel workers (default: 10)

# 02_extract.py options
--model MODEL           # Specific model to extract
--all                   # Extract for all models
--force                 # Force re-extraction
--extraction-model MODEL # Custom extraction model

# 03_metric.py options
--model MODEL           # Specific model for metrics
--all                   # Compute for all models
--compare               # Generate comparison plots
--models MODEL1,MODEL2  # Models to compare
```

### GPU and Performance Configuration

vLLM automatically manages GPU memory, but you can optimize for your setup via `code/config/config.py`:

- **GPU_MEMORY_UTILIZATION** (default: 0.9) - How much VRAM to use
- **MAX_MODEL_LENGTH** (default: 16384) - Maximum sequence length  
- **TENSOR_PARALLEL_SIZE** (default: 2) - Number of GPUs for tensor parallelism

For multi-GPU setups:
```bash
# Configure in code/config/config.py
TENSOR_PARALLEL_SIZE = 2  # Use 2 GPUs

# Or specify visible devices
export CUDA_VISIBLE_DEVICES=0,1
cd code/run && python 01_inference_vllm.py --model large_model
```

## Citation

If you use this evaluation pipeline, please cite the original SEAM benchmark paper:

```bibtex
@inproceedings{
tang2025seam,
title={{SEAM}: Semantically Equivalent Across Modalities Benchmark for Vision-Language Models},
author={Zhenwei Tang and Difan Jiao and Blair Yang and Ashton Anderson},
booktitle={Second Conference on Language Modeling},
year={2025},
url={https://openreview.net/forum?id=lI4LgGv4sX}
}
```