# SEAM: Semantically Equivalent Across Modalities Benchmark for Vision-Language Models

## Abstract

Evaluating whether vision–language models (VLMs) reason consistently across representations is challenging because modality comparisons are typically confounded by task differences and asymmetric information. We introduce **SEAM**, a benchmark that pairs semantically equivalent inputs across four domains with existing standardized textual and visual notations. By employing distinct notation systems across modalities, in contrast to OCR-based image-text pairing, SEAM provides a rigorous comparative assessment of the textual-symbolic and visual-spatial reasoning capabilities of VLMs. Across 16 contemporary models, we observe systematic modality imbalance: vision frequently lags language in overall performance, despite the problems containing semantically equivalent information, and cross-modal agreement is relatively low. Our error analysis reveals two main drivers: textual perception failures from tokenization in domain notations and visual perception failures that induce hallucinations. We also show that our results are largely robust to visual transformations. SEAM establishes a controlled, semantically equivalent setting for measuring and improving modality-agnostic reasoning.

## Overview

SEAM addresses fundamental limitations in existing benchmarks through its utilization of distinct notation systems and preservation of semantic equivalence across modalities. By leveraging domain-specific standardized representations in:

- **Chess**: Board images vs. FEN strings
- **Chemistry**: Structural diagrams vs. SMILES strings
- **Music**: Staff images vs. ABC notations
- **Graph Theory**: Node-edge diagrams vs. adjacency matrices

SEAM presents both visual-spatial and textual-symbolic representations while maintaining semantic equivalence. The benchmark comprises 16 carefully calibrated tasks designed to be self-contained in both modalities with 3,200 four-way multiple-choice questions in total.

## Features

- **vLLM Batch Inference**: Efficient offline batch processing using vLLM
- **Multi-Modal Support**: Language-only, vision-only, and vision-language evaluation
- **Automatic Answer Extraction**: Built-in LLM-based extraction with regex fallback
- **Progress Tracking**: Real-time progress bars with ETA using tqdm
- **Flexible Plotting**: Multiple plot types for comprehensive analysis
- **Model-Specific Organization**: Results saved in `results/{model_name}/` directories

## Setup

```bash
# Install dependencies and setup API keys
cd code/config && ./setup.sh

# Or manually:
cd code/config
pip install -r requirements.txt
cp api_keys.json.template api_keys.json
# Edit api_keys.json and add your API keys
```

### API Configuration

The project uses a secure JSON-based API key management system:

1. **Copy the template**: `cp api_keys.json.template api_keys.json`
2. **Edit api_keys.json**: Add your API keys:
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
3. **Security**: The `api_keys.json` file is automatically excluded from git via `.gitignore`

**Alternative**: You can still use environment variables if preferred:
```bash
export OPENAI_API_KEY=your_openai_key           # For OpenAI batch
export ANTHROPIC_API_KEY=your_anthropic_key     # For Claude batch
export HF_TOKEN=your_huggingface_token          # For gated HuggingFace models
```

## Quick Start

### Unified 3-Stage Pipeline

The SEAM benchmark now uses a unified 3-stage pipeline for all model types:

#### Stage 1: Inference
```bash
# vLLM (Local Models)
cd code/run && python 01_inference_vllm.py --model Qwen/Qwen2.5-VL-7B-Instruct --modes l,v,vl

# OpenAI (Real-time)
cd code/run && python 01_inference_openai.py --model gpt-4o-mini --modes l,v,vl

# OpenAI (Batch Processing)
cd code/run && python 01_inference_openai.py --model gpt-4o-mini --batch --action all --modes l,v,vl

# Claude (Real-time with parallel processing)
cd code/run && python 01_inference_claude.py --model claude-3-5-sonnet-20241022 --modes l,v,vl
```

#### Stage 2: Answer Extraction
```bash
# Extract for specific model
cd code/run && python 02_extract.py --model qwen-qwen2.5-vl-7b-instruct

# Extract for all models
cd code/run && python 02_extract.py --all
```

#### Stage 3: Metrics & Analysis
```bash
# Compute metrics for specific model
cd code/run && python 03_metric.py --model qwen-qwen2.5-vl-7b-instruct

# Generate comparison plots
cd code/run && python 03_metric.py --compare --models qwen-qwen2.5-vl-7b-instruct,gpt-4o-mini
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
# Custom GPU settings and specific tasks
cd code/run && python 01_inference_vllm.py --model Qwen/Qwen2.5-VL-7B-Instruct \
                                           --tasks fork,legal,puzzle \
                                           --gpu-memory-utilization 0.6 \
                                           --max-model-len 8192 \
                                           --tensor-parallel-size 1

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

## Dataset Preparation

### Option 1: Download from HuggingFace (Recommended)

The SEAM benchmark dataset is available on HuggingFace at [lilvjosephtang/SEAM-Benchmark](https://huggingface.co/datasets/lilvjosephtang/SEAM-Benchmark) and can be used automatically:

```bash
# The dataset will be automatically downloaded when running the evaluation pipeline
# No manual setup required - just run inference scripts directly

# For manual download using HuggingFace datasets:
python -c "
from datasets import load_dataset
dataset = load_dataset('lilvjosephtang/SEAM-Benchmark')
print('Dataset downloaded successfully')
print(f'Available tasks: {list(dataset.keys())}')
"
```

The HuggingFace dataset provides:
- **16 task-based splits** (fork, legal, puzzle, eval, carbon, hydrogen, weight, caption, notes, measures, forms, rhythm, path_counting, path_existence, shortest_path, bfs_traversal)
- **3,200 base samples** (200 samples per task)
- **Integrated images** stored as PIL Images for efficient loading
- **Rich metadata** including task domains, question types, and notation systems
- **Search functionality** through HuggingFace's interface

### Option 2: Download from Google Drive

You can download the pre-generated dataset from [this link](https://drive.google.com/drive/folders/12vruRWA56Sl4joIDH7uXF8QRmUcUoKwn?usp=sharing) and extract it to the `data/` directory.

### Option 3: Generate Dataset Manually

To generate the SEAM benchmark dataset manually, run the following scripts from the `code/dataset/` directory:

```bash
cd code/dataset/

# Generate Chemistry tasks
python dataset_chem.py

# Generate Chess tasks
python dataset_chess.py

# Generate Graph Theory tasks
python dataset_graph.py

# Generate Music tasks
python dataset_music.py
```

Each script will generate task-specific data, images, and question files in the `data/` directory.

### Generate Plots

```bash
# Basic plots (domains and heatmap)
./plot.sh

# Advanced comparison plots
python3 plot_comparison.py --plot-type all

# Specific plot types
python3 generate_plots.py --plot-type domains
python3 plot_comparison.py --plot-type task-heatmap --models InternVL3-8B InternVL3-14B
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

## Evaluation Details

### Tasks and Domains

The SEAM benchmark includes 16 tasks across 4 domains:

- **Chess**: fork, legal, puzzle, eval
- **Chemistry**: carbon, hydrogen, weight, caption
- **Music**: notes, measures, forms, rhythm
- **Graph**: path_counting, path_existence, shortest_path, bfs_traversal

### Modalities

Each task is evaluated in 3 modalities:
- **L (Language-only)**: Text-only input using standardized notations
- **V (Vision-only)**: Image-only input with visual representations
- **VL (Vision-Language)**: Combined text and image input

### Unified Output Schema

All inference scripts output results in a consistent JSON format:
```json
{
  "model": "qwen-qwen2.5-vl-7b-instruct",
  "task_name": "fork", "mode": "l", "index": 0,
  "question": "formatted_prompt", "answer": "C", "notation": "fen_string",
  "output": "model_response", "final_answer": "C", "correct": true,
  "latency": 2.1, "provider": "openai", "timestamp": "2025-08-12T21:30:00Z",
  "extraction_method": "llm", "extraction_confidence": 0.95
}
```

**Note**: The `latency` field represents actual per-sample processing time for OpenAI and Claude (real-time APIs), but is set to 0 for vLLM (batch inference) where individual sample timing is not meaningful.

### Answer Extraction

The unified extraction pipeline uses a two-stage process:
1. **LLM Extraction**: Primary method using Qwen2.5-7B-Instruct (temperature=0, max_tokens=5)
2. **Regex Fallback**: Pattern matching for common answer formats when LLM fails
3. **Confidence Scoring**: Tracks extraction method and confidence level

## Configuration

### Unified Configuration

All pipeline scripts use centralized configuration from `code/config/config.py`:
- **GPU Settings**: Memory utilization (default: 80%), tensor parallelism (default: 2 GPUs)
- **Model Parameters**: Temperature (0.7 inference, 0.0 extraction), max tokens (8192 inference, 5 extraction)
- **API Keys**: JSON-based configuration with environment variable fallback
- **Data Sources**: HuggingFace dataset (default) with JSONL fallback
- **Paths**: Benchmark data, results directory, model defaults
- **Tasks & Modes**: Complete task and modality definitions

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
--gpu-memory-utilization  # GPU memory utilization (default: 0.8)
--tensor-parallel-size   # Number of GPUs for tensor parallelism (default: 2)
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

## Handling GPU Memory and Performance

### Memory Management

vLLM automatically manages GPU memory, but you can optimize for your setup:

1. **GPU Memory Utilization**: Control how much VRAM to use
   ```bash
   # Use 70% of GPU memory instead of default 80%
   cd code/run && python run_vllm.py --model model_name --gpu-memory-utilization 0.7
   ```

2. **Model Length**: Reduce for memory-constrained setups
   ```bash
   # Reduce max sequence length to 8192 tokens
   cd code/run && python run_vllm.py --model model_name --max-model-len 8192
   ```

### Multi-GPU Support

For models that require multiple GPUs:
```bash
# Set tensor parallelism in code/config/config.py
DEFAULT_TENSOR_PARALLEL_SIZE = 2  # Use 2 GPUs

# Or specify visible devices
export CUDA_VISIBLE_DEVICES=0,1
cd code/run && python run_vllm.py --model large_model
```

### Performance Monitoring

```bash
# Monitor GPU usage during inference
watch -n1 nvidia-smi

# Check model loading progress
tail -f vllm_output.log
```

## Performance Optimization

- **Batch Processing**: vLLM automatically batches prompts for optimal throughput
- **GPU Utilization**: Configurable memory usage to maximize efficiency
- **Progress Tracking**: Real-time progress bars with throughput metrics
- **Automatic Caching**: Model weights cached for subsequent runs

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**: Reduce `--gpu-memory-utilization` or `--max-model-len`
2. **Model Loading Issues**: Check HuggingFace cache space and network connectivity
3. **Import Errors**: Ensure vLLM and transformers are properly installed
4. **Missing Images**: Ensure chess-bench dataset is properly downloaded
5. **InternVL Chat Template Issues**: The pipeline automatically handles InternVL's string-based format vs. other models' list-based format
6. **Noisy vLLM Warnings**: Tokenizer warnings are automatically suppressed for cleaner logs
7. **Gated Model Authentication**: For models like Llama that require HuggingFace tokens, add your token to `api_keys.json` or set `HF_TOKEN` environment variable

### Debug Mode

```bash
# Test with limited samples
cd code/run && python run_vllm.py --model model_name --debug-samples 5 --no-auto-extract

# Test single task and mode
cd code/run && python run_vllm.py --model model_name --tasks fork --modes l --debug-samples 2

# Manual answer extraction for debugging
cd code/run && python extract_answers.py --results-file ../../results/debug/results.jsonl
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