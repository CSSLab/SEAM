#!/usr/bin/env python3
"""
FIXED Task loader for SEAM benchmark evaluation.
Handles loading and formatting of all 16 tasks across 3 modalities with CORRECT image paths.
Supports both original JSONL files and HuggingFace parquet dataset.
"""

import json
import os
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class TaskLoader:
    """Fixed task loader that handles all task types with correct image paths"""
    
    def __init__(self, benchmark_root: str, use_hf_dataset: bool = None, hf_data_root: str = None):
        self.benchmark_root = Path(benchmark_root)
        
        # Define all tasks by domain FIRST
        self.chess_tasks = ["fork", "legal", "puzzle", "eval"]
        self.chem_tasks = ["carbon", "hydrogen", "weight", "caption"]
        self.music_tasks = ["notes", "measures", "forms", "rhythm"]
        self.graph_tasks = ["path_counting", "path_existence", "shortest_path", "bfs_traversal"]
        self.all_tasks = self.chess_tasks + self.chem_tasks + self.music_tasks + self.graph_tasks
        
        self.modes = ["l", "v", "vl"]
        
        # Import config here to avoid circular imports
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config.config import USE_HF_DATASET, HF_DATA_ROOT
            
            self.use_hf_dataset = use_hf_dataset if use_hf_dataset is not None else USE_HF_DATASET
            self.hf_data_root = Path(hf_data_root) if hf_data_root else HF_DATA_ROOT
        except ImportError:
            self.use_hf_dataset = False
            self.hf_data_root = None
            
        # Cache for temporary image files
        self._temp_image_cache = {}
        
        # Suffix for all tasks
        self.suffix = (
            "Let's think step-by-step to answer the above question.\n"
            "One and only one option is correct. If you are unsure, provide your best guess.\n"
            "If you believe none of the options are correct, select the closest one.\n"
            "You MUST conclude with: The best option is [the_option_letter].\n"
            "where the [the_option_letter] MUST be one of A, B, C or D.\n"
        )
    
    def cleanup_temp_images(self):
        """Clean up temporary image files."""
        for temp_path in self._temp_image_cache.values():
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                print(f"Warning: Could not remove temp file {temp_path}: {e}")
        
        self._temp_image_cache.clear()
        
        # Clean up temp directory if empty
        try:
            temp_dir = Path(tempfile.gettempdir()) / "seam_images"
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
        except Exception:
            pass  # Ignore cleanup errors
    
    def convert_modality_name(self, hf_modality: str) -> str:
        """Convert HF modality names to original format."""
        modality_map = {
            "Language": "l",
            "Vision": "v", 
            "Vision-Language": "vl"
        }
        return modality_map.get(hf_modality, hf_modality.lower())
    
    def convert_hf_to_original_format(self, task_name: str, hf_sample: dict) -> dict:
        """Convert HF dataset format to original JSONL format."""
        # Base conversion
        original_sample = {
            "_index": hf_sample.get("index", 0)
        }
        
        # Add notation fields based on task domain
        if task_name in self.chess_tasks:
            original_sample["fen"] = hf_sample.get("notation", "")
        elif task_name in self.chem_tasks:
            original_sample["smiles"] = hf_sample.get("notation", "")
        elif task_name in self.music_tasks:
            original_sample["abc_notation"] = hf_sample.get("notation", "")
        elif task_name in self.graph_tasks:
            # Convert string back to list/matrix if needed
            notation = hf_sample.get("notation", "")
            if notation and notation.startswith("[["):
                try:
                    original_sample["matrix"] = eval(notation)
                except:
                    original_sample["matrix"] = notation
            else:
                original_sample["matrix"] = notation
        
        # Convert options back to appropriate format
        options = self.get_options_for_task(task_name, hf_sample)
        if task_name == "legal":
            original_sample["options_uci"] = options
        elif task_name == "puzzle":
            original_sample["options_san"] = options
        else:
            original_sample["options"] = options
        
        # Add correct answer fields
        correct_idx = hf_sample.get("correct_idx", -1)
        correct_answer = hf_sample.get("correct_answer", "")
        
        if task_name == "fork":
            original_sample["correct_square"] = correct_answer
        elif task_name == "legal":
            original_sample["legal_move_idx"] = correct_idx
        elif task_name == "puzzle":
            original_sample["best_move_idx"] = correct_idx
        else:
            original_sample["correct_idx"] = correct_idx
        
        # Add task-specific fields
        if task_name in ["notes", "rhythm"]:
            # Extract from question or add default
            question = hf_sample.get("question", "")
            if "How many individual" in question:
                # Try to extract target note from question
                import re
                match = re.search(r"How many individual ([A-G])\b", question)
                if match:
                    original_sample["target_note"] = match.group(1)
                else:
                    # Fallback: extract from different pattern  
                    match = re.search(r"individual ([A-G])♯", question)
                    if match:
                        original_sample["target_note"] = match.group(1)
                    else:
                        original_sample["target_note"] = "G"  # Default fallback
            else:
                original_sample["target_note"] = "G"  # Default fallback
        
        if task_name == "rhythm":
            # Extract rhythm type from question or use default
            question = hf_sample.get("question", "")
            if "dotted sixteenth" in question:
                original_sample["rhythm_type"] = "dotted_sixteenth"
            elif "dotted eighth" in question:
                original_sample["rhythm_type"] = "dotted_eighth"
            elif "dotted half" in question:
                original_sample["rhythm_type"] = "dotted_half"
            else:
                original_sample["rhythm_type"] = "dotted_quarter"
        
        if task_name in self.graph_tasks:
            # Extract node information from question if available
            question = hf_sample.get("question", "")
            import re
            
            if task_name in ["path_counting", "path_existence", "shortest_path"]:
                # Extract source and target nodes
                match = re.search(r"from (\w+) to (\w+)", question)
                if match:
                    original_sample["source_node"] = match.group(1)
                    original_sample["target_node"] = match.group(2)
            elif task_name == "bfs_traversal":
                # Extract start node
                match = re.search(r"starting from node (\w+)", question)
                if match:
                    original_sample["start_node"] = match.group(1)
        
        return original_sample
    
    def get_options_for_task(self, task_name: str, hf_sample: dict) -> List[str]:
        """Convert HF option_a/b/c/d back to options list."""
        options = []
        for option_key in ["option_a", "option_b", "option_c", "option_d"]:
            option_value = hf_sample.get(option_key, "")
            if option_value:  # Only add non-empty options
                options.append(option_value)
        return options
    
    def save_pil_image_to_temp(self, pil_image: 'PILImage.Image', task_name: str, index: int) -> str:
        """Save PIL Image to temporary file and return path."""
        cache_key = f"{task_name}_{index}"
        
        # Check if already cached
        if cache_key in self._temp_image_cache:
            temp_path = self._temp_image_cache[cache_key]
            if os.path.exists(temp_path):
                return temp_path
        
        # Create temporary file
        temp_dir = Path(tempfile.gettempdir()) / "seam_images"
        temp_dir.mkdir(exist_ok=True)
        
        temp_path = temp_dir / f"{task_name}_{index}.png"
        pil_image.save(temp_path)
        
        # Cache the path
        self._temp_image_cache[cache_key] = str(temp_path)
        return str(temp_path)
    
    def format_options(self, options: List[str]) -> str:
        """Format multiple choice options"""
        prompt = f"A. {options[0]}\n"
        prompt += f"B. {options[1]}\n"
        prompt += f"C. {options[2]}\n"
        prompt += f"D. {options[3]}\n\n"
        return prompt
    
    def get_prefix(self, mode: str, notation: str, notation_name: str) -> str:
        """Get notation prefix based on mode"""
        if mode == "l" or mode == "vl":
            return f"{notation_name}: {notation}\n\n"
        elif mode == "v":
            return ""
        else:
            raise ValueError("Invalid mode")
    
    def load_task_data(self, task_name: str) -> List[dict]:
        """Load raw data for a task from HF parquet or original JSONL files"""
        if self.use_hf_dataset:
            return self.load_task_data_from_hf(task_name)
        else:
            return self.load_task_data_from_jsonl(task_name)
    
    def load_task_data_from_hf(self, task_name: str) -> List[dict]:
        """Load task data from HuggingFace parquet dataset."""
        if not HAS_PANDAS:
            print("Warning: pandas not available, falling back to JSONL")
            return self.load_task_data_from_jsonl(task_name)
            
        parquet_file = self.hf_data_root / f"{task_name}.parquet"
        if not parquet_file.exists():
            print(f"Warning: HF parquet file not found: {parquet_file}")
            print(f"Falling back to JSONL data")
            return self.load_task_data_from_jsonl(task_name)
        
        try:
            # Load parquet file
            df = pd.read_parquet(parquet_file)
            print(f"📊 Loaded {len(df)} samples from HF dataset: {task_name}")
            
            # HF dataset now contains only base samples (200 per task)
            # No need to filter - each row is a unique base sample
            base_samples = df.sort_values('index')
            
            print(f"📊 Using {len(base_samples)} base samples for task generation")
            
            # Convert to original format
            data = []
            for _, row in base_samples.iterrows():
                hf_sample = row.to_dict()
                original_sample = self.convert_hf_to_original_format(task_name, hf_sample)
                
                # Handle image if present
                if hf_sample.get("image") is not None and HAS_PIL:
                    try:
                        image_data = hf_sample["image"]
                        
                        # Check if image is PIL Image object or needs conversion
                        if hasattr(image_data, 'save'):
                            # Already a PIL Image
                            pil_image = image_data
                        elif isinstance(image_data, dict) and 'bytes' in image_data:
                            # Image stored as bytes in dict (common in HF datasets)
                            from io import BytesIO
                            pil_image = PILImage.open(BytesIO(image_data['bytes']))
                        elif isinstance(image_data, dict) and 'path' in image_data:
                            # Image stored as path reference
                            pil_image = PILImage.open(image_data['path'])
                        else:
                            # Skip image conversion for now
                            continue
                        
                        # Save PIL image to temp file for compatibility
                        temp_path = self.save_pil_image_to_temp(
                            pil_image, 
                            task_name, 
                            hf_sample.get("index", len(data))
                        )
                        original_sample["_temp_image_path"] = temp_path
                    except Exception as e:
                        # For now, just skip images that can't be processed
                        pass
                
                data.append(original_sample)
            
            return data
            
        except Exception as e:
            print(f"Warning: Error reading HF parquet file {parquet_file}: {e}")
            print(f"Falling back to JSONL data")
            return self.load_task_data_from_jsonl(task_name)
    
    def load_task_data_from_jsonl(self, task_name: str) -> List[dict]:
        """Load raw data for a task from original JSONL files"""
        task_file = self.benchmark_root / f"{task_name}.jsonl"
        if not task_file.exists():
            print(f"Warning: Task file not found: {task_file}")
            return []
        
        data = []
        try:
            with open(task_file, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        # Add index information for music and graph tasks
                        item['_index'] = line_num  # 0-based index matching image names
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Error parsing line {line_num+1} in {task_file}: {e}")
                        continue
        except Exception as e:
            print(f"Warning: Error reading task file {task_file}: {e}")
            return []
        
        return data
    
    def format_chess_task(self, task_name: str, data_item: dict, mode: str) -> Tuple[str, str, str]:
        """Format chess tasks (fork, legal, puzzle, eval)"""
        fen = data_item["fen"]
        
        if task_name == "fork":
            answer = chr(data_item["options"].index(data_item["correct_square"]) + 65)
            instructions = (
                "Question: Which of the following pieces is forking other pieces in this chess position?\n"
                "A forking piece is one that attacks two or more enemy pieces simultaneously.\n"
            )
            prompt = self.get_prefix(mode, fen, "FEN") + instructions + self.format_options(data_item["options"]) + self.suffix
        
        elif task_name == "legal":
            answer = chr(data_item["legal_move_idx"] + 65)
            instructions = "Question: Which of the following moves is legal in this chess position?\n"
            prompt = self.get_prefix(mode, fen, "FEN") + instructions + self.format_options(data_item["options_uci"]) + self.suffix
        
        elif task_name == "puzzle":
            answer = chr(data_item["best_move_idx"] + 65)
            instructions = "Question: Which of the following moves is the best next move for the active player in this chess position?\n"
            prompt = self.get_prefix(mode, fen, "FEN") + instructions + self.format_options(data_item["options_san"]) + self.suffix
        
        elif task_name == "eval":
            answer = chr(data_item["correct_idx"] + 65)
            instructions = (
                "Question: Which of the following is the correct centipawn evaluation for this chess position?\n"
                "The evaluation shows the overall positional strength (including piece activity, king safety, and other strategic factors), not just the material count and piece value advantage.\n"
                "Positive numbers mean White has an advantage, negative numbers mean Black has an advantage, and 0 means the position is equal.\n"
            )
            prompt = self.get_prefix(mode, fen, "FEN") + instructions + self.format_options(data_item["options"]) + self.suffix
        
        else:
            raise ValueError(f"Unknown chess task: {task_name}")
        
        return prompt, answer, fen
    
    def format_chem_task(self, task_name: str, data_item: dict, mode: str) -> Tuple[str, str, str]:
        """Format chemistry tasks (carbon, hydrogen, weight, caption)"""
        smiles = data_item["smiles"]
        answer = chr(data_item["correct_idx"] + 65)
        
        if task_name == "carbon":
            instructions = "Question: Which of the following is the correct number of carbon atoms in this compound?\n"
        elif task_name == "hydrogen":
            instructions = "Question: Which of the following is the correct number of hydrogen atoms in this compound?\n"
        elif task_name == "weight":
            instructions = "Question: Which of the following is the correct molecular weight of this compound?\n"
        elif task_name == "caption":
            instructions = "Question: Which of the following descriptions is correct for this compound?\n"
        else:
            raise ValueError(f"Unknown chemistry task: {task_name}")
        
        prompt = self.get_prefix(mode, smiles, "SMILES") + instructions + self.format_options(data_item["options"]) + self.suffix
        return prompt, answer, smiles
    
    def format_music_task(self, task_name: str, data_item: dict, mode: str) -> Tuple[str, str, str]:
        """Format music tasks (notes, measures, forms, rhythm)"""
        abc = data_item["abc_notation"]
        answer = chr(data_item["correct_idx"] + 65)
        
        if task_name == "notes":
            instructions = (
                f"Question: How many individual {data_item['target_note']} notes (including {data_item['target_note']}♯, "
                f"{data_item['target_note']}♭, and {data_item['target_note']}♮) appear in this piece? Count all occurrences "
                f"of {data_item['target_note']} regardless of whether they are sharp, flat, or natural, but do not count "
                f"notes that appear as part of chords.\n"
            )
        elif task_name == "measures":
            instructions = (
                "Question: Which of the following is the correct number of measures in this piece?\n"
                "Count each measure only once, ignoring any repetition signs in the score.\n"
            )
        elif task_name == "forms":
            instructions = "Question: Which of the following best describes the musical form of this piece?\n"
        elif task_name == "rhythm":
            pattern_names = {
                "dotted_sixteenth": "dotted sixteenth note",
                "dotted_eighth": "dotted eighth note",
                "dotted_quarter": "dotted quarter note",
                "dotted_half": "dotted half note"
            }
            instructions = f"Question: Which of the following measures contains a {pattern_names[data_item['rhythm_type']]}?\n"
        else:
            raise ValueError(f"Unknown music task: {task_name}")
        
        prompt = self.get_prefix(mode, abc, "ABC Notation") + instructions + self.format_options(data_item["options"]) + self.suffix
        return prompt, answer, abc
    
    def format_graph_task(self, task_name: str, data_item: dict, mode: str) -> Tuple[str, str, str]:
        """Format graph tasks (path_counting, path_existence, shortest_path, bfs_traversal)"""
        matrix = data_item["matrix"]
        answer = chr(data_item["correct_idx"] + 65)
        
        if task_name == "path_counting":
            source_node = data_item["source_node"]
            target_node = data_item["target_node"]
            instructions = f"Question: Which of the following is the correct number of unique simple paths from {source_node} to {target_node} in the graph?\n"
        
        elif task_name == "path_existence":
            source_node = data_item["source_node"]
            target_node = data_item["target_node"]
            instructions = f"Question: Which of the following node lists represents a path from {source_node} to {target_node} in the graph?\n"
        
        elif task_name == "shortest_path":
            source_node = data_item["source_node"]
            target_node = data_item["target_node"]
            instructions = f"Question: Which of the following is the length of the shortest simple path from {source_node} to {target_node} in the graph?\n"
        
        elif task_name == "bfs_traversal":
            start_node = data_item['start_node']
            instructions = f"Question: Which of the following node lists represents the order of the BFS traversal starting from node {start_node} in the graph?\n"
        
        else:
            raise ValueError(f"Unknown graph task: {task_name}")
        
        prompt = self.get_prefix(mode, matrix, "Adjacency matrix") + instructions + self.format_options(data_item["options"]) + self.suffix
        return prompt, answer, matrix
    
    def format_task(self, task_name: str, data_item: dict, mode: str) -> Tuple[str, str, str]:
        """Format any task and return prompt, answer, and notation"""
        
        if task_name in self.chess_tasks:
            prompt, answer, notation = self.format_chess_task(task_name, data_item, mode)
        
        elif task_name in self.chem_tasks:
            prompt, answer, notation = self.format_chem_task(task_name, data_item, mode)
        
        elif task_name in self.music_tasks:
            prompt, answer, notation = self.format_music_task(task_name, data_item, mode)
        
        elif task_name in self.graph_tasks:
            prompt, answer, notation = self.format_graph_task(task_name, data_item, mode)
        
        else:
            raise ValueError(f"Unknown task: {task_name}")
        
        return prompt, answer, notation
    
    def get_image_path(self, task_name: str, data_item: dict) -> Optional[str]:
        """Get image path for a task sample"""
        # Check if we have a temporary image path from HF dataset
        if "_temp_image_path" in data_item:
            temp_path = data_item["_temp_image_path"]
            if os.path.exists(temp_path):
                return temp_path
        
        # Fall back to original image path logic
        if task_name in self.chess_tasks:
            fen = data_item["fen"]
            return self._get_chess_image_path(fen, task_name)
        
        elif task_name in self.chem_tasks:
            smiles = data_item["smiles"]
            return self._get_chem_image_path(smiles, task_name)
        
        elif task_name in self.music_tasks:
            data_index = data_item.get('_index', None)
            return self._get_music_image_path(data_index, task_name)
        
        elif task_name in self.graph_tasks:
            data_index = data_item.get('_index', None)
            return self._get_graph_image_path(data_index, task_name)
        
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    def get_vision_only_prompt(self, task_name: str, data_item: dict) -> str:
        """Get vision-only prompt for a task sample (no notation prefix)"""
        # Get the full prompt first
        full_prompt, _, _ = self.format_task(task_name, data_item, "l")
        
        # Extract just the instruction and options part (remove notation prefix)
        # Find the "Question:" part which starts the actual task
        question_start = full_prompt.find("Question:")
        if question_start != -1:
            return full_prompt[question_start:]
        else:
            # Fallback: return the full prompt
            return full_prompt
    
    def _hash_str(self, text: str) -> str:
        """Hash a string using MD5 (same as original utils.py)"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_chess_image_path(self, fen: str, task_name: str) -> str:
        """Get chess board image path using FEN notation"""
        # Based on utils.py read_image_chess function:
        # fen.split(" ")[0].replace("/", "_") + ".png"
        fen_board = fen.split(" ")[0].replace("/", "_")
        image_filename = f"{fen_board}.png"
        image_path = self.benchmark_root / task_name / image_filename
        
        if image_path.exists():
            return str(image_path)
        else:
            print(f"Warning: Chess image not found: {image_path}")
            return self._get_fallback_image()
    
    def _get_chem_image_path(self, smiles: str, task_name: str) -> str:
        """Get chemistry image path using SMILES hash"""
        # Based on utils.py read_image_chem function:
        # hash_str(smiles) + ".png"
        hash_filename = f"{self._hash_str(smiles)}.png"
        image_path = self.benchmark_root / task_name / hash_filename
        
        if image_path.exists():
            return str(image_path)
        else:
            print(f"Warning: Chemistry image not found: {image_path}")
            return self._get_fallback_image()
    
    def _get_music_image_path(self, index: Optional[int], task_name: str) -> str:
        """Get music image path using index"""
        # Based on utils.py read_image_music function:
        # f"{index}.png"
        if index is None:
            print(f"Warning: No index provided for music task {task_name}")
            return self._get_fallback_image()
            
        image_filename = f"{index}.png"
        image_path = self.benchmark_root / task_name / image_filename
        
        if image_path.exists():
            return str(image_path)
        else:
            print(f"Warning: Music image not found: {image_path}")
            return self._get_fallback_image()
    
    def _get_graph_image_path(self, index: Optional[int], task_name: str) -> str:
        """Get graph image path using index"""
        # Based on utils.py read_image_graph function:
        # f"{idx}.png"
        if index is None:
            print(f"Warning: No index provided for graph task {task_name}")
            return self._get_fallback_image()
            
        image_filename = f"{index}.png"
        image_path = self.benchmark_root / task_name / image_filename
        
        if image_path.exists():
            return str(image_path)
        else:
            print(f"Warning: Graph image not found: {image_path}")
            return self._get_fallback_image()
    
    def _get_fallback_image(self) -> str:
        """Get fallback test image"""
        test_image = "test.jpg"
        if os.path.exists(test_image):
            return test_image
        # Create a minimal 1x1 pixel image if test.jpg doesn't exist
        try:
            from PIL import Image
            img = Image.new('RGB', (1, 1), color='white')
            img.save(test_image)
            return test_image
        except:
            return test_image  # Return anyway, let the error handle downstream