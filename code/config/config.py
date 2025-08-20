#!/usr/bin/env python3
"""
Configuration settings for SEAM benchmark vLLM batch inference.
"""

from pathlib import Path

# Path configuration - absolute paths
BENCHMARK_ROOT = Path("/datadrive/josephtang/SEAM/code/data")  # Original JSONL data (fallback)
HF_DATA_ROOT = Path("/datadrive/josephtang/SEAM/data")  # HuggingFace parquet data
RESULTS_DIR = Path("/datadrive/josephtang/SEAM/results")

# Data source configuration
USE_HF_DATASET = True  # Use HuggingFace parquet dataset by default
HF_REPO_ID = "lilvjosephtang/SEAM-Benchmark"  # HuggingFace repository ID

# vLLM Batch inference configuration
MAX_MODEL_LENGTH = 16384  # Maximum model sequence length
GPU_MEMORY_UTILIZATION = 0.9  # GPU memory utilization for vLLM
TENSOR_PARALLEL_SIZE = 2  # Number of GPUs for tensor parallelism
PIPELINE_PARALLEL_SIZE = 1  # Number of GPUs for pipeline parallelism
MAX_NUM_SEQS = 256  # Maximum number of sequences processed concurrently (default: 256, reduce for large vision models)

# Model defaults
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
EXTRACTION_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# System prompts
MODEL_SYSTEM_PROMPT = "You are a helpful assistant."
EXTRACTION_SYSTEM_PROMPT = "You are a helpful final answer extractor."

# Temperature settings
EXTRACTION_TEMPERATURE = 0.0  # For deterministic answer extraction

# Generation settings
MODEL_MAX_TOKENS = 8192  # Max new tokens for generation
EXTRACTION_MAX_TOKENS = 5  # Max tokens for answer extraction

# Legacy Benchmark settings (removed, now using functions above)

# Task and mode definitions
ALL_TASKS = [
    # Chess tasks
    "fork", "legal", "puzzle", "eval",
    # Chemistry tasks  
    "carbon", "hydrogen", "weight", "caption",
    # Music tasks
    "notes", "measures", "forms", "rhythm",
    # Graph theory tasks
    "path_counting", "path_existence", "shortest_path", "bfs_traversal"
]

ALL_MODES = ["l", "v", "vl"]  # language-only, vision-only, vision-language

# File extensions and formats
RESULTS_FORMAT = "jsonl"  # results file format
CSV_SUMMARY = True  # generate CSV summaries
AUTO_EXTRACT = True  # automatically run answer extraction