#!/usr/bin/env python3
"""
FIXED Task loader for SEAM benchmark evaluation.
Handles loading and formatting of all 16 tasks across 3 modalities with CORRECT image paths.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class TaskLoader:
    """Fixed task loader that handles all task types with correct image paths"""
    
    def __init__(self, benchmark_root: str):
        self.benchmark_root = Path(benchmark_root)
        
        # Define all tasks by domain
        self.chess_tasks = ["fork", "legal", "puzzle", "eval"]
        self.chem_tasks = ["carbon", "hydrogen", "weight", "caption"]
        self.music_tasks = ["notes", "measures", "forms", "rhythm"]
        self.graph_tasks = ["path_counting", "path_existence", "shortest_path", "bfs_traversal"]
        self.all_tasks = self.chess_tasks + self.chem_tasks + self.music_tasks + self.graph_tasks
        
        self.modes = ["l", "v", "vl"]
        
        # Suffix for all tasks
        self.suffix = (
            "Let's think step-by-step to answer the above question.\n"
            "One and only one option is correct. If you are unsure, provide your best guess.\n"
            "If you believe none of the options are correct, select the closest one.\n"
            "You MUST conclude with: The best option is [the_option_letter].\n"
            "where the [the_option_letter] MUST be one of A, B, C or D.\n"
        )
    
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
        """Load raw data for a task and add index information"""
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