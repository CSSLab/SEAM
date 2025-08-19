# SEAM: Semantically Equivalent Across Modalities Benchmark for Vision-Language Models

## About SEAM

Evaluating whether vision–language models (VLMs) reason consistently across representations is challenging because modality comparisons are typically confounded by task differences and asymmetric information. We introduce **SEAM**, a benchmark that pairs semantically equivalent inputs across four domains with existing standardized textual and visual notations. By employing distinct notation systems across modalities, in contrast to OCR-based image-text pairing, SEAM provides a rigorous comparative assessment of the textual-symbolic and visual-spatial reasoning capabilities of VLMs. Across 16 contemporary models, we observe systematic modality imbalance: vision frequently lags language in overall performance, despite the problems containing semantically equivalent information, and cross-modal agreement is relatively low. Our error analysis reveals two main drivers: textual perception failures from tokenization in domain notations and visual perception failures that induce hallucinations. We also show that our results are largely robust to visual transformations. SEAM establishes a controlled, semantically equivalent setting for measuring and improving modality-agnostic reasoning.

## Dataset Structure

The benchmark data is organized directly in this directory with the following structure:

- **JSONL files**: Each task has a corresponding `.jsonl` file containing the questions and answers
- **Image directories**: Each task has an associated directory containing the visual representations
- **16 Tasks across 4 domains**:
  - **Chess**: fork, legal, puzzle, eval
  - **Chemistry**: carbon, hydrogen, weight, caption
  - **Music**: notes, measures, forms, rhythm  
  - **Graph Theory**: path_counting, path_existence, shortest_path, bfs_traversal

## Data Format

Each JSONL file contains multiple-choice questions with the following structure:
- Question text with standardized notation (for language-only mode)
- Image path reference (for vision and vision-language modes)
- Four answer choices (A, B, C, D)
- Correct answer
- Associated metadata

## Usage

This data is used by the SEAM evaluation pipeline to test vision-language models across three modalities:
- **L (Language-only)**: Text-based questions using standardized notations
- **V (Vision-only)**: Image-based questions with visual representations
- **VL (Vision-Language)**: Combined text and image inputs

## Download and Setup

### Option 1: HuggingFace Datasets (Recommended)

The SEAM benchmark dataset is available on HuggingFace and can be downloaded automatically:

```bash
# Install the datasets library if not already installed
pip install datasets

# Download the dataset
python -c "
from datasets import load_dataset
dataset = load_dataset('seam-benchmark/SEAM')
print('Dataset downloaded successfully')
"
```

The dataset will be automatically cached and available for use with the evaluation pipeline.

### Option 2: Google Drive Download

You can download the pre-generated dataset from [this link](https://drive.google.com/drive/folders/12vruRWA56Sl4joIDH7uXF8QRmUcUoKwn?usp=sharing) and extract it to this directory.

### Option 3: Generate from Source

Alternatively, generate the data using the scripts in `../code/dataset/`:

```bash
cd ../code/dataset/
python dataset_chess.py
python dataset_chem.py  
python dataset_music.py
python dataset_graph.py
```

## Citation

If you use this dataset, please cite:

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