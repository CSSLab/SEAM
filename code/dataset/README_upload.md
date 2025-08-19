# SEAM Dataset Upload to HuggingFace Hub

This directory contains the script to upload the SEAM benchmark dataset to HuggingFace Hub.

## Quick Start

### 1. Install Requirements

```bash
pip install datasets huggingface_hub pillow
```

### 2. Authenticate with HuggingFace

Either set your HF token in the API keys config:

```json
# code/config/api_keys.json
{
  "huggingface": {
    "api_key": "your_huggingface_write_token_here"
  }
}
```

Or set environment variable:

```bash
export HF_TOKEN=your_huggingface_write_token_here
```

### 3. Upload Dataset

```bash
# Upload to public repository
python upload_to_huggingface.py --repo-id username/seam-benchmark

# Upload to private repository
python upload_to_huggingface.py --repo-id username/seam-benchmark --private

# Test processing without uploading
python upload_to_huggingface.py --repo-id username/seam-benchmark --dry-run
```

## Dataset Structure

The uploaded dataset will have the following structure:

### Splits
- `all`: Complete dataset with all modalities
- `l`: Language-only samples  
- `v`: Vision-only samples
- `vl`: Vision-Language samples

### Sample Format

Each sample contains:

```python
{
    "task": "carbon",                    # Task name
    "domain": "chemistry",               # Domain category
    "modality": "vl",                   # Modality (l/v/vl)
    "index": 0,                         # Sample index
    "question": "How many carbon atoms...", # Question text
    "notation": "COC(=O)CCC...",        # Standardized notation
    "notation_type": "SMILES",          # Notation type
    "options": ["13", "16", "19", "22"], # Multiple choice options
    "correct_answer": "16",             # Correct answer
    "correct_idx": 1,                   # Index of correct option
    "image": <PIL.Image>                # Associated image (if applicable)
}
```

### Domains and Tasks

#### Chess Domain
- **Tasks**: fork, legal, puzzle, eval
- **Notation**: FEN (Forsyth-Edwards Notation)
- **Images**: Chess board positions

#### Chemistry Domain  
- **Tasks**: carbon, hydrogen, weight, caption
- **Notation**: SMILES (Simplified Molecular Input Line Entry System)
- **Images**: Chemical structure diagrams

#### Music Domain
- **Tasks**: notes, measures, forms, rhythm  
- **Notation**: ABC notation
- **Images**: Musical staff notation

#### Graph Theory Domain
- **Tasks**: path_counting, path_existence, shortest_path, bfs_traversal
- **Notation**: Adjacency matrices
- **Images**: Graph node-edge diagrams

## Usage Examples

### Loading the Dataset

```python
from datasets import load_dataset

# Load all data
dataset = load_dataset("username/seam-benchmark")

# Load specific modality
language_only = load_dataset("username/seam-benchmark", split="l")
vision_only = load_dataset("username/seam-benchmark", split="v") 
vision_language = load_dataset("username/seam-benchmark", split="vl")

# Filter by domain
chemistry_samples = dataset["all"].filter(lambda x: x["domain"] == "chemistry")

# Filter by task
carbon_samples = dataset["all"].filter(lambda x: x["task"] == "carbon")
```

### Evaluation Example

```python
# Evaluate model on vision-language modality
vl_dataset = load_dataset("username/seam-benchmark", split="vl")

for sample in vl_dataset:
    # Format prompt based on modality and notation
    if sample["modality"] == "vl":
        prompt = f"Question: {sample['question']}\nOptions: {sample['options']}"
        image = sample["image"]
        
        # Run your VLM here
        response = model.generate(prompt, image)
        
        # Check if response matches correct_answer
        correct = extract_answer(response) == sample["correct_answer"]
```

## Script Features

- **Automatic Authentication**: Uses HF tokens from config or environment
- **Robust Image Handling**: Properly loads and encodes images for HuggingFace
- **Comprehensive Metadata**: Includes domain, task, and modality information
- **Error Handling**: Graceful handling of missing files or data
- **Progress Tracking**: Clear progress indicators during upload
- **Dry Run Mode**: Test processing without uploading

## Troubleshooting

### Common Issues

1. **Authentication Error**: Ensure your HF token has write permissions
2. **Missing Images**: Check that all image files exist in the expected directories
3. **Memory Issues**: The dataset contains many images; ensure sufficient RAM
4. **Network Timeout**: Large uploads may take time; ensure stable connection

### HuggingFace Token Permissions

Your token needs:
- **Write** access to create/update datasets
- **Read** access to verify uploads

### File Structure Requirements

The script expects this structure:
```
data/
├── carbon.jsonl
├── carbon/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── fork.jsonl
├── fork/
│   ├── 0.png
│   └── ...
└── ...
```

## Dataset Statistics

- **Total Samples**: 9,600 (16 tasks × 200 samples × 3 modalities)
- **Domains**: 4 (Chess, Chemistry, Music, Graph Theory)
- **Tasks**: 16 total (4 per domain)
- **Modalities**: 3 (Language-only, Vision-only, Vision-Language)
- **Image Format**: PNG
- **Text Encoding**: UTF-8

## License

The SEAM benchmark dataset is released under the MIT License.