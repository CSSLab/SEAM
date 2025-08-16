#!/usr/bin/env python3
"""
Upload SEAM benchmark dataset to HuggingFace Hub.

This script uploads the SEAM (Semantically Equivalent Across Modalities) benchmark
dataset to HuggingFace Hub with proper structure and metadata.

Usage:
    python upload_to_huggingface.py --repo-id your-username/seam-benchmark [--private]

Requirements:
    pip install datasets huggingface_hub pillow
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import datasets
    from datasets import Dataset, DatasetDict, Features, Value, Image
    from huggingface_hub import HfApi, login
    from PIL import Image as PILImage
except ImportError as e:
    print(f"❌ Missing required packages. Please install:")
    print("pip install datasets huggingface_hub pillow")
    sys.exit(1)

# Add parent directory to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import BENCHMARK_ROOT, ALL_TASKS

class SEAMDatasetUploader:
    """Upload SEAM benchmark dataset to HuggingFace Hub."""
    
    def __init__(self, repo_id: str, private: bool = False):
        self.repo_id = repo_id
        self.private = private
        self.api = HfApi()
        self.benchmark_root = Path(BENCHMARK_ROOT)
        
        # Domain mapping
        self.task_domains = {
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
        
        # Modality descriptions
        self.modality_descriptions = {
            "l": "Language-only (text using standardized notations)",
            "v": "Vision-only (images without text)",
            "vl": "Vision-Language (combined text and images)"
        }
        
    def authenticate(self) -> bool:
        """Authenticate with HuggingFace Hub."""
        try:
            # Try to load HF token from config
            config_path = Path(__file__).parent.parent / "config" / "api_keys.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    hf_token = config.get('huggingface', {}).get('api_key')
                    if hf_token:
                        login(token=hf_token)
                        print("🔑 Authenticated with HuggingFace using config token")
                        return True
            
            # Try environment variable
            if os.getenv('HF_TOKEN'):
                login(token=os.getenv('HF_TOKEN'))
                print("🔑 Authenticated with HuggingFace using environment token")
                return True
                
            # Interactive login
            print("⚠️  No HF token found. Please login interactively:")
            login()
            return True
            
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    
    def load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def fen_to_filename(self, fen: str) -> str:
        """Convert FEN string to filesystem-safe filename using the exact same logic as dataset generation."""
        # Use the same logic as dataset_chess.py: fen.split(" ")[0].replace("/", "_") + ".png"
        filename = fen.split(" ")[0].replace("/", "_") + ".png"
        return filename
    
    def get_image_name_for_task(self, task: str, index: int, data_item: Dict[str, Any]) -> str:
        """Get the correct image filename for a task based on its naming pattern."""
        domain = self.task_domains.get(task, "unknown")
        
        # Chess tasks: Use FEN-based naming
        if domain == "chess" and "fen" in data_item:
            return self.fen_to_filename(data_item["fen"])
        
        # Chemistry tasks: Use hash-based naming (check if available in data)
        elif domain == "chemistry":
            # Check if there's an image hash/filename in the data
            if "image_file" in data_item:
                return data_item["image_file"]
            elif "smiles" in data_item:
                # For chemistry, images might be named by SMILES hash
                # We'll need to find the actual file in the directory
                return self.find_image_by_index(task, index)
        
        # Graph and Music tasks: Use sequential numbering
        else:
            return f"{index}.png"
    
    def find_image_by_index(self, task: str, index: int) -> str:
        """Find the actual image file for chemistry tasks by matching the index."""
        image_dir = self.benchmark_root / task
        if image_dir.exists():
            # Get all PNG files and sort them
            png_files = sorted([f for f in image_dir.iterdir() if f.suffix == '.png'])
            if index < len(png_files):
                return png_files[index].name
        return f"{index}.png"  # Fallback to sequential naming
    
    def get_image_path(self, task: str, image_name: str) -> Optional[Path]:
        """Get path to image file for a task."""
        image_dir = self.benchmark_root / task
        if image_dir.exists():
            image_path = image_dir / image_name
            if image_path.exists():
                return image_path
        return None
    
    def create_sample(self, task: str, modality: str, index: int, 
                     data_item: Dict[str, Any]) -> Dict[str, Any]:
        """Create a single dataset sample."""
        domain = self.task_domains.get(task, "unknown")
        
        # Base sample structure
        sample = {
            "task": task,
            "domain": domain,
            "modality": modality,
            "index": index,
            "question_type": "multiple_choice",
            "options": data_item.get("options", []),
            "correct_answer": data_item["options"][data_item["correct_idx"]] 
                             if "correct_idx" in data_item else None,
            "correct_idx": data_item.get("correct_idx", -1)
        }
        
        # Add task-specific data
        if "smiles" in data_item:
            sample["notation"] = data_item["smiles"]
            sample["notation_type"] = "SMILES"
        elif "fen" in data_item:
            sample["notation"] = data_item["fen"]
            sample["notation_type"] = "FEN"
        elif "abc" in data_item:
            sample["notation"] = data_item["abc"]
            sample["notation_type"] = "ABC"
        elif "matrix" in data_item:
            sample["notation"] = str(data_item["matrix"])
            sample["notation_type"] = "adjacency_matrix"
        else:
            sample["notation"] = None
            sample["notation_type"] = None
        
        # Add question text if available
        sample["question"] = data_item.get("question", "")
        
        # Handle images based on modality
        if modality in ["v", "vl"]:
            image_name = self.get_image_name_for_task(task, index, data_item)
            image_path = self.get_image_path(task, image_name)
            if image_path and image_path.exists():
                # Load image directly - datasets will handle encoding
                sample["image"] = str(image_path)
            else:
                print(f"⚠️  Warning: Image not found for {task}/{modality}/{index} (expected: {image_name})")
                sample["image"] = None
        else:
            sample["image"] = None
            
        return sample
    
    def process_task(self, task: str) -> List[Dict[str, Any]]:
        """Process all samples for a given task."""
        samples = []
        task_file = self.benchmark_root / f"{task}.jsonl"
        
        if not task_file.exists():
            print(f"⚠️  Warning: Task file not found: {task_file}")
            return samples
        
        print(f"📁 Processing task: {task}")
        data = self.load_jsonl(task_file)
        
        for modality in ["l", "v", "vl"]:
            for index, data_item in enumerate(data):
                sample = self.create_sample(task, modality, index, data_item)
                samples.append(sample)
        
        print(f"✅ Processed {len(samples)} samples for task {task}")
        return samples
    
    def create_dataset_features(self) -> Features:
        """Define the dataset schema/features."""
        return Features({
            "task": Value("string"),
            "domain": Value("string"), 
            "modality": Value("string"),
            "index": Value("int32"),
            "question_type": Value("string"),
            "question": Value("string"),
            "notation": Value("string"),
            "notation_type": Value("string"),
            "options": datasets.Sequence(Value("string")),
            "correct_answer": Value("string"),
            "correct_idx": Value("int32"),
            "image": Image()
        })
    
    def create_dataset_card(self) -> str:
        """Create README content for the dataset."""
        return f"""---
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
size_categories:
- 1K<n<10K
---

# SEAM: Semantically Equivalent Across Modalities Benchmark

## Dataset Description

SEAM (Semantically Equivalent Across Modalities) is a high-performance Vision-Language Model evaluation benchmark that tests models across 4 domains with 16 tasks and 3 modalities.

### Key Features

- **4 Domains**: Chess, Chemistry, Music, Graph Theory  
- **16 Tasks**: 4 tasks per domain
- **3 Modalities**: Language-only (L), Vision-only (V), Vision-Language (VL)
- **3,200 Questions**: 200 questions × 16 tasks, each evaluated in 3 modalities = 9,600 total evaluations

### Domain-Specific Representations

- **Chess**: Board images vs. FEN strings
- **Chemistry**: Structural diagrams vs. SMILES strings  
- **Music**: Staff notation vs. ABC notation
- **Graph Theory**: Node-edge diagrams vs. adjacency matrices

### Tasks by Domain

#### Chess Domain
- `fork`: Chess fork detection
- `legal`: Legal move validation
- `puzzle`: Chess puzzle solving
- `eval`: Position evaluation

#### Chemistry Domain  
- `carbon`: Carbon atom counting
- `hydrogen`: Hydrogen atom counting
- `weight`: Molecular weight calculation
- `caption`: Chemical structure description

#### Music Domain
- `notes`: Note identification
- `measures`: Measure counting
- `forms`: Musical form recognition
- `rhythm`: Rhythm pattern analysis

#### Graph Theory Domain
- `path_counting`: Path counting problems
- `path_existence`: Path existence queries
- `shortest_path`: Shortest path finding
- `bfs_traversal`: BFS traversal order

### Modalities

- **L (Language-only)**: Text input using standardized domain notations
- **V (Vision-only)**: Image input with visual representations
- **VL (Vision-Language)**: Combined text and image input

### Dataset Structure

Each sample contains:
- `task`: Task name (e.g., "fork", "carbon")
- `domain`: Domain category (chess, chemistry, music, graph)
- `modality`: Evaluation modality (l, v, vl)
- `index`: Sample index within task
- `question`: Question text (if applicable)
- `notation`: Standardized notation (FEN, SMILES, ABC, matrix)
- `notation_type`: Type of notation used
- `options`: Multiple choice options
- `correct_answer`: Correct answer
- `correct_idx`: Index of correct option
- `image`: Associated image (for v and vl modalities)

### Citation

If you use this dataset, please cite the original SEAM benchmark paper.

### License

This dataset is released under the MIT License.
"""
    
    def upload_dataset(self) -> bool:
        """Upload the complete dataset to HuggingFace Hub."""
        try:
            print("🚀 Starting SEAM dataset upload to HuggingFace Hub")
            print(f"📍 Repository: {self.repo_id}")
            print(f"🔒 Private: {self.private}")
            
            # Authenticate
            if not self.authenticate():
                return False
                
            # Process all tasks
            all_samples = []
            for task in ALL_TASKS:
                task_samples = self.process_task(task)
                all_samples.extend(task_samples)
            
            if not all_samples:
                print("❌ No samples found to upload")
                return False
                
            print(f"📊 Total samples: {len(all_samples)}")
            
            # Create dataset
            print("🔄 Creating HuggingFace dataset...")
            features = self.create_dataset_features()
            dataset = Dataset.from_list(all_samples, features=features)
            
            # Split by modality for easier access
            splits = {}
            for modality in ["l", "v", "vl"]:
                modality_samples = [s for s in all_samples if s["modality"] == modality]
                if modality_samples:
                    splits[modality] = Dataset.from_list(modality_samples, features=features)
            
            # Create combined dataset dict
            dataset_dict = DatasetDict({
                "all": dataset,
                **splits
            })
            
            # Upload to Hub
            print("⬆️  Uploading to HuggingFace Hub...")
            dataset_dict.push_to_hub(
                self.repo_id,
                private=self.private,
                commit_message="Upload SEAM benchmark dataset"
            )
            
            # Create and upload dataset card
            print("📝 Creating dataset card...")
            readme_content = self.create_dataset_card()
            self.api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                repo_type="dataset"
            )
            
            print("✅ Dataset uploaded successfully!")
            print(f"🔗 Dataset URL: https://huggingface.co/datasets/{self.repo_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Upload SEAM dataset to HuggingFace Hub")
    parser.add_argument("--repo-id", required=True, 
                       help="HuggingFace repository ID (e.g., username/seam-benchmark)")
    parser.add_argument("--private", action="store_true",
                       help="Make the dataset private")
    parser.add_argument("--dry-run", action="store_true",
                       help="Process data without uploading")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🧪 Dry run mode - processing data without uploading")
        uploader = SEAMDatasetUploader(args.repo_id, args.private)
        
        # Test processing all tasks
        total_samples = 0
        missing_images = 0
        for task in ALL_TASKS:
            print(f"\n📁 Processing task: {task}")
            samples = uploader.process_task(task)
            task_missing = sum(1 for s in samples if s.get('image') is None and s.get('modality') in ['v', 'vl'])
            missing_images += task_missing
            total_samples += len(samples)
            print(f"✅ Task {task}: {len(samples)} samples ({task_missing} missing images)")
            if samples:
                print(f"   Sample structure: {list(samples[0].keys())}")
        
        print(f"\n📊 Summary:")
        print(f"   Total tasks: {len(ALL_TASKS)}")
        print(f"   Total samples: {total_samples}")
        print(f"   Missing images: {missing_images}")
        print(f"   Expected total: {len(ALL_TASKS) * 200 * 3}")
        return
    
    # Validate repo ID format
    if "/" not in args.repo_id:
        print("❌ Error: repo-id must be in format 'username/dataset-name'")
        sys.exit(1)
    
    # Create uploader and upload
    uploader = SEAMDatasetUploader(args.repo_id, args.private)
    success = uploader.upload_dataset()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()