#!/usr/bin/env python3
"""
SEAM dataset upload to HuggingFace Hub with 16 task-based splits.
Ensures images are properly displayed on HuggingFace website.
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict

try:
    from datasets import Dataset, DatasetDict, Features, Value, Image, Sequence
    from huggingface_hub import HfApi
    from PIL import Image as PILImage
except ImportError as e:
    print(f"❌ Missing packages: {e}")
    print("Install: pip install datasets huggingface_hub pillow")
    exit(1)

# Add parent directory to path for config imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import ALL_TASKS

def load_hf_token():
    """Load HuggingFace token from config or environment."""
    config_path = Path(__file__).parent.parent / "config" / "api_keys.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            hf_token = config.get('huggingface', {}).get('api_key')
            if hf_token:
                return hf_token
    
    import os
    return os.getenv('HF_TOKEN')

def load_jsonl(file_path):
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def get_image_path(task, index, data_item, benchmark_root):
    """Get image path for a task sample using SEAM's naming convention."""
    import hashlib
    
    image_dir = benchmark_root / task
    if not image_dir.exists():
        return None
    
    # Chess tasks use FEN-based naming
    if task in ["fork", "legal", "puzzle", "eval"]:
        fen = data_item.get("fen", "")
        if fen:
            fen_board = fen.split(" ")[0].replace("/", "_")
            image_path = image_dir / f"{fen_board}.png"
            if image_path.exists():
                return image_path
    
    # Chemistry tasks use SMILES hash naming
    elif task in ["carbon", "hydrogen", "weight", "caption"]:
        smiles = data_item.get("smiles", "")
        if smiles:
            smiles_hash = hashlib.md5(smiles.encode()).hexdigest()
            image_path = image_dir / f"{smiles_hash}.png"
            if image_path.exists():
                return image_path
    
    # Music and graph tasks use index-based naming
    elif task in ["notes", "measures", "forms", "rhythm", "path_counting", "path_existence", "shortest_path", "bfs_traversal"]:
        image_path = image_dir / f"{index}.png"
        if image_path.exists():
            return image_path
    
    return None

def get_question_text(task, data_item):
    """Generate question text for a task based on SEAM format."""
    # Chess tasks
    if task == "fork":
        return "Which of the following pieces is forking other pieces in this chess position? A forking piece is one that attacks two or more enemy pieces simultaneously."
    elif task == "legal":
        return "Which of the following moves is legal in this chess position?"
    elif task == "puzzle":
        return "Which of the following moves is the best next move for the active player in this chess position?"
    elif task == "eval":
        return ("Which of the following is the correct centipawn evaluation for this chess position? "
                "The evaluation shows the overall positional strength (including piece activity, king safety, and other strategic factors), "
                "not just the material count and piece value advantage. Positive numbers mean White has an advantage, "
                "negative numbers mean Black has an advantage, and 0 means the position is equal.")
    
    # Chemistry tasks
    elif task == "carbon":
        return "Which of the following is the correct number of carbon atoms in this compound?"
    elif task == "hydrogen":
        return "Which of the following is the correct number of hydrogen atoms in this compound?"
    elif task == "weight":
        return "Which of the following is the correct molecular weight of this compound?"
    elif task == "caption":
        return "Which of the following descriptions is correct for this compound?"
    
    # Music tasks
    elif task == "notes":
        target_note = data_item.get('target_note', 'target')
        return (f"How many individual {target_note} notes (including {target_note}♯, "
                f"{target_note}♭, and {target_note}♮) appear in this piece? Count all occurrences "
                f"of {target_note} regardless of whether they are sharp, flat, or natural, but do not count "
                f"notes that appear as part of chords.")
    elif task == "measures":
        return "Which of the following is the correct number of measures in this piece? Count each measure only once, ignoring any repetition signs in the score."
    elif task == "forms":
        return "Which of the following best describes the musical form of this piece?"
    elif task == "rhythm":
        pattern_names = {
            "dotted_sixteenth": "dotted sixteenth note",
            "dotted_eighth": "dotted eighth note",
            "dotted_quarter": "dotted quarter note",
            "dotted_half": "dotted half note"
        }
        rhythm_type = data_item.get('rhythm_type', 'dotted_quarter')
        pattern_name = pattern_names.get(rhythm_type, rhythm_type)
        return f"Which of the following measures contains a {pattern_name}?"
    
    # Graph tasks
    elif task == "path_counting":
        source = data_item.get('source_node', 'source')
        target = data_item.get('target_node', 'target')
        return f"Which of the following is the correct number of unique simple paths from {source} to {target} in the graph?"
    elif task == "path_existence":
        source = data_item.get('source_node', 'source')
        target = data_item.get('target_node', 'target')
        return f"Which of the following node lists represents a path from {source} to {target} in the graph?"
    elif task == "shortest_path":
        source = data_item.get('source_node', 'source')
        target = data_item.get('target_node', 'target')
        return f"Which of the following is the length of the shortest simple path from {source} to {target} in the graph?"
    elif task == "bfs_traversal":
        start = data_item.get('start_node', 'start')
        return f"Which of the following node lists represents the order of the BFS traversal starting from node {start} in the graph?"
    
    return ""

def create_base_sample(task, index, data_item, benchmark_root):
    """Create a base dataset sample with only essential data (no modality-specific fields)."""
    # Domain mapping
    task_domains = {
        # Chess domain
        "fork": "chess", "legal": "chess", "puzzle": "chess", "eval": "chess",
        # Chemistry domain  
        "carbon": "chemistry", "hydrogen": "chemistry", "weight": "chemistry", 
        "caption": "chemistry",
        # Music domain
        "notes": "music", "measures": "music", "forms": "music", "rhythm": "music",
        # Graph theory domain
        "path_counting": "graph", "path_existence": "graph", 
        "shortest_path": "graph", "bfs_traversal": "graph"
    }
    
    domain = task_domains.get(task, "unknown")
    
    # Handle different correct answer field names across tasks
    correct_idx = -1
    correct_answer = ""
    
    # Get the correct options field based on task
    if task == "legal":
        options = data_item.get("options_uci", [])
    elif task == "puzzle":
        options = data_item.get("options_san", [])
    else:
        # fork, eval, chemistry, music, graph tasks use "options"
        options = data_item.get("options", [])
    
    if "correct_idx" in data_item:
        # Standard format (chemistry, music, graph, eval tasks)
        correct_idx = data_item["correct_idx"]
        if 0 <= correct_idx < len(options):
            correct_answer = str(options[correct_idx])
    elif "legal_move_idx" in data_item:
        # Legal chess task (uses options_uci)
        correct_idx = data_item["legal_move_idx"]
        if 0 <= correct_idx < len(options):
            correct_answer = str(options[correct_idx])
    elif "best_move_idx" in data_item:
        # Puzzle chess task (uses options_san)
        correct_idx = data_item["best_move_idx"]
        if 0 <= correct_idx < len(options):
            correct_answer = str(options[correct_idx])
    elif "correct_square" in data_item:
        # Fork chess task (uses options)
        correct_answer = data_item["correct_square"]
        if correct_answer in options:
            correct_idx = options.index(correct_answer)
    
    # Ensure we have exactly 4 options, pad with empty strings if needed
    padded_options = [str(opt) for opt in options] + [""] * (4 - len(options))
    
    sample = {
        "task": task,
        "domain": domain,
        "index": index,
        "question_type": "multiple_choice",
        "option_a": padded_options[0],
        "option_b": padded_options[1], 
        "option_c": padded_options[2],
        "option_d": padded_options[3],
        "correct_answer": correct_answer,
        "correct_idx": correct_idx
    }
    
    # Add task-specific notation
    if "smiles" in data_item:
        sample["notation"] = data_item["smiles"]
        sample["notation_type"] = "SMILES"
    elif "fen" in data_item:
        sample["notation"] = data_item["fen"]
        sample["notation_type"] = "FEN"
    elif "abc_notation" in data_item:
        sample["notation"] = data_item["abc_notation"]
        sample["notation_type"] = "ABC"
    elif "matrix" in data_item:
        sample["notation"] = str(data_item["matrix"])
        sample["notation_type"] = "adjacency_matrix"
    else:
        sample["notation"] = ""
        sample["notation_type"] = ""
    
    # Add question text using SEAM's format
    sample["question"] = get_question_text(task, data_item)
    
    # Load and store the image for this sample
    image_path = get_image_path(task, index, data_item, benchmark_root)
    if image_path and image_path.exists():
        try:
            sample["image"] = PILImage.open(image_path)
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            sample["image"] = None
    else:
        sample["image"] = None
        
    return sample

def create_sample(task, modality, index, data_item, benchmark_root):
    """Create a single dataset sample."""
    # Domain mapping
    task_domains = {
        # Chess domain
        "fork": "chess", "legal": "chess", "puzzle": "chess", "eval": "chess",
        # Chemistry domain  
        "carbon": "chemistry", "hydrogen": "chemistry", "weight": "chemistry", 
        "caption": "chemistry",
        # Music domain
        "notes": "music", "measures": "music", "forms": "music", "rhythm": "music",
        # Graph theory domain
        "path_counting": "graph", "path_existence": "graph", 
        "shortest_path": "graph", "bfs_traversal": "graph"
    }
    
    domain = task_domains.get(task, "unknown")
    
    # Handle different correct answer field names and option fields across tasks
    correct_idx = -1
    correct_answer = ""
    
    # Get the correct options field based on task
    if task == "legal":
        options = data_item.get("options_uci", [])
    elif task == "puzzle":
        options = data_item.get("options_san", [])
    else:
        # fork, eval, chemistry, music, graph tasks use "options"
        options = data_item.get("options", [])
    
    if "correct_idx" in data_item:
        # Standard format (chemistry, music, graph, eval tasks)
        correct_idx = data_item["correct_idx"]
        if 0 <= correct_idx < len(options):
            correct_answer = str(options[correct_idx])
    elif "legal_move_idx" in data_item:
        # Legal chess task (uses options_uci)
        correct_idx = data_item["legal_move_idx"]
        if 0 <= correct_idx < len(options):
            correct_answer = str(options[correct_idx])
    elif "best_move_idx" in data_item:
        # Puzzle chess task (uses options_san)
        correct_idx = data_item["best_move_idx"]
        if 0 <= correct_idx < len(options):
            correct_answer = str(options[correct_idx])
    elif "correct_square" in data_item:
        # Fork chess task (uses options)
        correct_answer = data_item["correct_square"]
        if correct_answer in options:
            correct_idx = options.index(correct_answer)
    
    # Base sample structure
    # Map modality codes to full names
    modality_names = {
        "l": "Language",
        "v": "Vision", 
        "vl": "Vision-Language"
    }
    
    # Ensure we have exactly 4 options, pad with empty strings if needed
    padded_options = [str(opt) for opt in options] + [""] * (4 - len(options))
    
    sample = {
        "task": task,
        "domain": domain,
        "modality": modality_names.get(modality, modality),
        "index": index,
        "question_type": "multiple_choice",
        "option_a": padded_options[0],
        "option_b": padded_options[1], 
        "option_c": padded_options[2],
        "option_d": padded_options[3],
        "correct_answer": correct_answer,
        "correct_idx": correct_idx
    }
    
    # Add task-specific notation
    if "smiles" in data_item:
        sample["notation"] = data_item["smiles"]
        sample["notation_type"] = "SMILES"
    elif "fen" in data_item:
        sample["notation"] = data_item["fen"]
        sample["notation_type"] = "FEN"
    elif "abc_notation" in data_item:
        sample["notation"] = data_item["abc_notation"]
        sample["notation_type"] = "ABC"
    elif "matrix" in data_item:
        sample["notation"] = str(data_item["matrix"])
        sample["notation_type"] = "adjacency_matrix"
    else:
        sample["notation"] = ""
        sample["notation_type"] = ""
    
    # Add question text using SEAM's format
    sample["question"] = get_question_text(task, data_item)
    
    # Handle images - load as PIL Image for proper HF display
    if modality in ["v", "vl"]:
        image_path = get_image_path(task, index, data_item, benchmark_root)
        if image_path and image_path.exists():
            sample["image"] = PILImage.open(image_path)
        else:
            sample["image"] = None
    else:
        # Explicitly set to None for Language-only samples
        sample["image"] = None
        
    return sample

def create_dataset_features():
    """Define the dataset schema for base samples (no modality field)."""
    return Features({
        "task": Value("string"),
        "domain": Value("string"), 
        "index": Value("int32"),
        "question_type": Value("string"),
        "question": Value("string"),
        "notation": Value("string"),
        "notation_type": Value("string"),
        "option_a": Value("string"),
        "option_b": Value("string"), 
        "option_c": Value("string"),
        "option_d": Value("string"),
        "correct_answer": Value("string"),
        "correct_idx": Value("int32"),
        "image": Image()
    })

def create_dataset_card():
    """Create dataset card content based on SEAM website."""
    return """---
license: mit
task_categories:
- visual-question-answering
- multiple-choice
language:
- en
tags:
- vision-language
- multimodal
- benchmark
- chess
- chemistry
- music
- graph-theory
- semantic-equivalence
- VLM
size_categories:
- 1K<n<10K
dataset_info:
  features:
  - name: task
    dtype: string
  - name: domain
    dtype: string
  - name: index
    dtype: int32
  - name: question_type
    dtype: string
  - name: question
    dtype: string
  - name: notation
    dtype: string
  - name: notation_type
    dtype: string
  - name: option_a
    dtype: string
  - name: option_b
    dtype: string
  - name: option_c
    dtype: string
  - name: option_d
    dtype: string
  - name: correct_answer
    dtype: string
  - name: correct_idx
    dtype: int32
  - name: image
    dtype: image
  splits:
  - name: fork
    num_bytes: 0
    num_examples: 200
  - name: legal
    num_bytes: 0
    num_examples: 200
  - name: puzzle
    num_bytes: 0
    num_examples: 200
  - name: eval
    num_bytes: 0
    num_examples: 200
  - name: carbon
    num_bytes: 0
    num_examples: 200
  - name: hydrogen
    num_bytes: 0
    num_examples: 200
  - name: weight
    num_bytes: 0
    num_examples: 200
  - name: caption
    num_bytes: 0
    num_examples: 200
  - name: notes
    num_bytes: 0
    num_examples: 200
  - name: measures
    num_bytes: 0
    num_examples: 200
  - name: forms
    num_bytes: 0
    num_examples: 200
  - name: rhythm
    num_bytes: 0
    num_examples: 200
  - name: path_counting
    num_bytes: 0
    num_examples: 200
  - name: path_existence
    num_bytes: 0
    num_examples: 200
  - name: shortest_path
    num_bytes: 0
    num_examples: 200
  - name: bfs_traversal
    num_bytes: 0
    num_examples: 200
  download_size: 0
  dataset_size: 0
configs:
- config_name: default
  data_files:
  - split: fork
    path: data/fork-*
  - split: legal
    path: data/legal-*
  - split: puzzle
    path: data/puzzle-*
  - split: eval
    path: data/eval-*
  - split: carbon
    path: data/carbon-*
  - split: hydrogen
    path: data/hydrogen-*
  - split: weight
    path: data/weight-*
  - split: caption
    path: data/caption-*
  - split: notes
    path: data/notes-*
  - split: measures
    path: data/measures-*
  - split: forms
    path: data/forms-*
  - split: rhythm
    path: data/rhythm-*
  - split: path_counting
    path: data/path_counting-*
  - split: path_existence
    path: data/path_existence-*
  - split: shortest_path
    path: data/shortest_path-*
  - split: bfs_traversal
    path: data/bfs_traversal-*
---

# SEAM: Semantically Equivalent Across Modalities Benchmark for Vision-Language Models

*[CSSLab](https://csslab.cs.toronto.edu/), Department of Computer Science, University of Toronto*  
*[COLM '25] Second Conference on Language Modeling*

- **Paper**: [OpenReview](https://openreview.net/pdf?id=lI4LgGv4sX)
- **Leaderboard**: [SEAM Benchmark](https://lilv98.github.io/SEAM-Website/)
- **Code**: [GitHub](https://github.com/CSSLab/SEAM)

![Overview](https://lilv98.github.io/SEAM-Website/static/images/main.png)

## Abstract

Evaluating whether vision–language models (VLMs) reason consistently across representations is challenging because modality comparisons are typically confounded by task differences and asymmetric information. We introduce **SEAM**, a benchmark that pairs semantically equivalent inputs across four domains with existing standardized textual and visual notations. By employing distinct notation systems across modalities, in contrast to OCR-based image-text pairing, SEAM provides a rigorous comparative assessment of the textual-symbolic and visual-spatial reasoning capabilities of VLMs. Across 21 contemporary models, we observe systematic modality imbalance: vision frequently lags language in overall performance, despite the problems containing semantically equivalent information, and cross-modal agreement is relatively low. Our error analysis reveals two main drivers: textual perception failures from tokenization in domain notations and visual perception failures that induce hallucinations. We also show that our results are largely robust to visual transformations. SEAM establishes a controlled, semantically equivalent setting for measuring and improving modality-agnostic reasoning.

## Key Features

- **4 Domains**: Chess, Chemistry, Music, Graph Theory with standardized notations
- **16 Tasks**: 4 tasks per domain (64 total task-modality combinations)
- **3 Modalities**: Language-only (L), Vision-only (V), Vision-Language (VL)
- **3,200 Base Samples**: 200 samples × 16 tasks
- **9,600 Evaluations**: TaskLoader generates 3 modality-specific prompts per base sample
- **Semantic Equivalence**: Same information presented in different representational formats

## Domains and Notation Systems

### Chess Domain
- **Tasks**: `fork`, `legal`, `puzzle`, `eval`
- **Textual**: FEN (Forsyth-Edwards Notation)
- **Visual**: Chess board diagrams

### Chemistry Domain  
- **Tasks**: `carbon`, `hydrogen`, `weight`, `caption`
- **Textual**: SMILES (Simplified Molecular Input Line Entry System)
- **Visual**: Chemical structure diagrams

### Music Domain
- **Tasks**: `notes`, `measures`, `forms`, `rhythm`
- **Textual**: ABC notation
- **Visual**: Musical staff notation

### Graph Theory Domain
- **Tasks**: `path_counting`, `path_existence`, `shortest_path`, `bfs_traversal`
- **Textual**: Adjacency matrices
- **Visual**: Node-edge diagrams

## Dataset Splits

The dataset is organized into 16 task-based splits (600 samples each):

- **Chess**: `fork`, `legal`, `puzzle`, `eval`
- **Chemistry**: `carbon`, `hydrogen`, `weight`, `caption`  
- **Music**: `notes`, `measures`, `forms`, `rhythm`
- **Graph Theory**: `path_counting`, `path_existence`, `shortest_path`, `bfs_traversal`

Each split contains 200 base samples. TaskLoader generates modality-specific prompts (L, V, VL) from these base samples.

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("lilvjosephtang/SEAM-Benchmark")

# Access specific tasks
chess_fork = dataset["fork"]  # Chess fork detection (600 samples)
chemistry_carbon = dataset["carbon"]  # Carbon atom counting (600 samples)

# Each task contains 200 base samples
# TaskLoader generates modality-specific prompts (L/V/VL) from these base samples
print(f"Task {chess_fork[0]['task']} has {len(chess_fork)} base samples")

# Example sample structure
sample = chess_fork[0]
print(f"Task: {sample['task']}")
print(f"Domain: {sample['domain']}")
# No modality field - TaskLoader handles modality generation
print(f"Question: {sample['question']}")
print(f"Options: A) {sample['option_a']}, B) {sample['option_b']}, C) {sample['option_c']}, D) {sample['option_d']}")
print(f"Correct Answer: {sample['correct_answer']}")
print(f"Notation: {sample['notation']}")  # FEN string for chess
# sample['image'] contains the chess board image for Vision/Vision-Language modalities
```

## Sample Structure

Each sample contains:
- `task`: Task identifier (e.g., "fork", "carbon")
- `domain`: Domain category ("chess", "chemistry", "music", "graph")
- No modality field (TaskLoader generates modality-specific prompts)
- `index`: Sample index within the task
- `question`: Question text (if applicable)
- `notation`: Domain-specific notation (FEN, SMILES, ABC, adjacency matrix)
- `notation_type`: Type of notation used
- `option_a`, `option_b`, `option_c`, `option_d`: Multiple choice options
- `correct_answer`: The correct answer
- `correct_idx`: Index of the correct option
- `image`: Associated image (PIL Image, None for base storage - TaskLoader handles image loading for V/VL modalities)

## Evaluation Protocol

SEAM enables three types of evaluation:

1. **Language**: Models receive only textual notation
2. **Vision**: Models receive only visual representation  
3. **Vision-Language**: Models receive both notation and image

The semantic equivalence across modalities allows for direct comparison of reasoning capabilities and cross-modal agreement analysis.


## Citation

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

"""

def clean_old_files(repo_id, hf_token):
    """Clean up old files from previous uploads with batch deletion."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        
        # Get list of files in repo
        try:
            repo_files = api.list_repo_files(repo_id, repo_type="dataset")
            print(f"📋 Found {len(repo_files)} existing files")
            
            # Files to keep (we'll regenerate these)
            keep_files = {".gitattributes", "README.md"}
            
            # Delete old files in batch
            files_to_delete = [f for f in repo_files if f not in keep_files]
            if files_to_delete:
                print(f"🗑️  Cleaning up {len(files_to_delete)} old files in batch...")
                
                # Group files by type for better commit messages
                parquet_files = [f for f in files_to_delete if f.endswith('.parquet')]
                other_files = [f for f in files_to_delete if not f.endswith('.parquet')]
                
                # Delete parquet files in one commit
                if parquet_files:
                    for file_path in parquet_files:
                        try:
                            api.delete_file(
                                path_in_repo=file_path,
                                repo_id=repo_id,
                                repo_type="dataset",
                                commit_message="Clean up old dataset files"
                            )
                        except Exception as e:
                            print(f"⚠️  Could not delete {file_path}: {e}")
                
                # Delete other files in another commit
                if other_files:
                    for file_path in other_files:
                        try:
                            api.delete_file(
                                path_in_repo=file_path,
                                repo_id=repo_id,
                                repo_type="dataset",
                                commit_message="Clean up old metadata files"
                            )
                        except Exception as e:
                            print(f"⚠️  Could not delete {file_path}: {e}")
                            
                print("✅ Cleanup complete")
            else:
                print("✅ No old files to clean")
                
        except Exception as e:
            print(f"⚠️  Could not list repo files: {e}")
            print("📝 Proceeding with upload anyway...")
            
    except Exception as e:
        print(f"⚠️  Cleanup failed: {e}")
        print("📝 Proceeding with upload anyway...")

def upload_to_hf(args):
    """Upload SEAM dataset with 16 task-based splits."""
    print(f"🚀 Creating SEAM dataset with 16 task splits")
    print(f"📍 Repository: {args.repo_id}")
    print(f"📁 Data path: {args.folder_path}")
    
    benchmark_root = Path(args.folder_path)
    if not benchmark_root.exists():
        print(f"❌ Data directory not found: {benchmark_root}")
        return
    
    # Load HF token
    hf_token = load_hf_token()
    if not hf_token:
        print("❌ No HuggingFace token found")
        return
    
    # Clean up old files
    clean_old_files(args.repo_id, hf_token)
    
    # Process each task
    task_datasets = {}
    
    for task in ALL_TASKS:
        print(f"📁 Processing task: {task}")
        task_file = benchmark_root / f"{task}.jsonl"
        
        if not task_file.exists():
            print(f"⚠️  Task file not found: {task_file}")
            continue
        
        # Load task data
        data = load_jsonl(task_file)
        task_samples = []
        
        # Create base samples without modality variations (TaskLoader will handle modalities)
        for index, data_item in enumerate(data):
            sample = create_base_sample(task, index, data_item, benchmark_root)
            task_samples.append(sample)
        
        # Create dataset for this task
        if task_samples:
            features = create_dataset_features()
            task_datasets[task] = Dataset.from_list(task_samples, features=features)
            print(f"✅ Task {task}: {len(task_samples)} samples")
    
    if not task_datasets:
        print("❌ No datasets created")
        return
    
    # Create DatasetDict with 16 task splits
    dataset_dict = DatasetDict(task_datasets)
    
    print(f"📊 Created {len(task_datasets)} task splits")
    print(f"📋 Splits: {list(dataset_dict.keys())}")
    
    # Upload to HuggingFace
    print("⬆️  Uploading to HuggingFace Hub...")
    try:
        dataset_dict.push_to_hub(
            args.repo_id,
            token=hf_token,
            commit_message="Upload SEAM benchmark with 16 task-based splits"
        )
        
        # Upload dataset card
        print("📝 Creating dataset card...")
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        readme_content = create_dataset_card()
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add dataset card"
        )
        
        print("✅ Upload complete!")
        print(f"🔗 https://huggingface.co/datasets/{args.repo_id}")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")

def download_from_hf(args):
    """Download dataset from HuggingFace Hub."""
    print(f"🚀 Downloading {args.repo_id}")
    
    # Setup cache directory
    cache_dir = Path("/datadrive/josephtang/SEAM/data/hf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
        
        # Load dataset with custom cache
        hf_token = load_hf_token()
        dataset = load_dataset(args.repo_id, token=hf_token, cache_dir=str(cache_dir))
        print(f"📋 Available splits: {list(dataset.keys())}")
        
        # Save to local directory
        local_path = Path(args.output_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Save each split as parquet
        for split_name, split_data in dataset.items():
            split_file = local_path / f"{split_name}.parquet"
            split_data.to_parquet(str(split_file))
            print(f"💾 Saved {split_name}: {len(split_data)} samples to {split_file}")
        
        print(f"✅ Download complete! Data saved to {args.output_dir}")
        
    finally:
        # Clean up cache
        if cache_dir.exists():
            print(f"🗑️  Cleaning up cache: {cache_dir}")
            shutil.rmtree(cache_dir)

def upload_card_only(args):
    """Upload only the dataset card/README."""
    print(f"📝 Uploading dataset card to {args.repo_id}")
    
    # Load HF token
    hf_token = load_hf_token()
    if not hf_token:
        print("❌ No HuggingFace token found")
        return
    
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        readme_content = create_dataset_card()
        
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Update dataset card"
        )
        
        print("✅ Dataset card uploaded!")
        print(f"🔗 https://huggingface.co/datasets/{args.repo_id}")
        
    except Exception as e:
        print(f"❌ Card upload failed: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="SEAM dataset management")
    parser.add_argument("--repo-id", default="lilvjosephtang/SEAM-Benchmark",
                       help="HuggingFace repository ID")
    parser.add_argument("--folder-path", default="../../raw_data",
                       help="Path to data folder")
    parser.add_argument("--output-dir", default="../../data",
                       help="Output directory for downloads")
    parser.add_argument("--stage", default="upload", choices=["upload", "download", "card-only"],
                       help="Stage to run: upload (full dataset), download, or card-only (README only)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.stage == "upload":
        upload_to_hf(args)
    elif args.stage == "download":
        download_from_hf(args)
    elif args.stage == "card-only":
        upload_card_only(args)