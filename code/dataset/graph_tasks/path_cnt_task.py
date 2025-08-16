import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm
from tqdm import trange
import json
import os
import math
from collections import defaultdict
from graph_tasks.config import *
from graph_tasks.base_tasks import GraphImageGenerator

class PathCountingTask:
    def __init__(self):
        self.source_node = None
        self.target_node = None
        self.correct_answer = None
        self.offset_answers = []
        self.task_type = "path_counting"

    def generate_random_graph(self):
        max_attempts = 50
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            # Step 1: Randomly generate an undirected graph G
            num_nodes = random.randint(NODE_LIMITS['path_counting'][0], NODE_LIMITS['path_counting'][1])
            # num_nodes = random.randint(NODE_LIMIT[0], NODE_LIMIT[1])
            # Adjust edge probability to get more interesting path counts
            edge_probability = min(2 / num_nodes, 0.5)  # Slightly lower density for more paths
            G = nx.gnp_random_graph(num_nodes, edge_probability, directed=False)
            
            # Check if the graph is connected
            if not nx.is_connected(G):
                continue
                
            # Choose random source and target nodes
            nodes = list(G.nodes())
            if len(nodes) < 2:
                continue
                
            # Try different source-target pairs to find one with multiple paths
            source_target_pairs = []
            for src in nodes:
                for tgt in nodes:
                    if src != tgt:
                        source_target_pairs.append((src, tgt))
                        
            random.shuffle(source_target_pairs)
            
            for src, tgt in source_target_pairs:
                self.source_node = src
                self.target_node = tgt
                
                # Check if path exists in G from src to tgt
                if not nx.has_path(G, src, tgt):
                    continue
                
                # Try to count all simple paths
                # Use a reasonable cutoff to prevent excessive computation
                try:
                    # Using a cutoff helps prevent exponential computation
                    # We limit to paths of reasonable length (≤ 8 edges)
                    paths = list(nx.all_simple_paths(G, src, tgt, cutoff=8))
                    path_count = len(paths)
                    
                    # We want questions with more than one path (preferably)
                    # But not too many paths which would be computationally expensive
                    if path_count > 1 and path_count < 10:
                        return G, path_count
                        
                except nx.NetworkXError:
                    # Handle any NetworkX errors (e.g., excessive computation)
                    continue
            
        print(f"Failed to generate suitable graph after {max_attempts} attempts")
        return None, None

    def generate_options(self, G, path_count):
        """
        Generate the options for the path counting task.
        Returns the correct answer and a list of offset answers.
        """
        if path_count is None:
            return None, None
        
        # Calculate edge count for offset calculation
        edge_count = G.number_of_edges()
        offset = math.ceil(0.1 * edge_count + 1)
        
        # Correct answer is the path count
        self.correct_answer = path_count
        
        # Generate offset options (correct answer ± offset)
        offset_options = []
        
        # Add offset answers, ensuring they're positive
        for delta in range(-offset, offset + 1):
            if delta != 0:  # Skip the correct answer
                candidate = path_count + delta
                if candidate > 0:  # Avoid negative or zero values
                    offset_options.append(candidate)
        
        # If we don't have enough options, add more by increasing the offset
        while len(offset_options) < 3:
            offset += 1
            candidate_plus = path_count + offset
            candidate_minus = path_count - offset
            
            if candidate_minus > 0 and candidate_minus not in offset_options:
                offset_options.append(candidate_minus)
                
            if len(offset_options) < 3 and candidate_plus not in offset_options:
                offset_options.append(candidate_plus)
        
        # Ensure we have exactly 3 offset options
        if len(offset_options) > 3:
            # Prioritize keeping a balanced distribution around the correct answer
            # Sort by absolute difference from correct answer
            offset_options.sort(key=lambda x: abs(x - path_count))
            offset_options = offset_options[:3]
        
        self.offset_answers = offset_options
        return self.correct_answer, self.offset_answers

    def update_jsonl(self, matrix, options, source_node, target_node, correct_idx, jsonl_path):
        """
        Update a JSONL file with new data about path counting.
        
        Args:
            matrix (list): The adjacency matrix of the undirected graph.
            options (list): The path count options.
            source_node (int): The source node.
            target_node (int): The target node.
            correct_idx (int): The index of the correct count.
        """
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        
        # Prepare the data entry
        entry = {
            "matrix": matrix,
            "options": options,
            "source_node": source_node,
            "target_node": target_node,
            "correct_idx": correct_idx
        }
        
        # Append the new entry to the JSONL file
        with open(jsonl_path, 'a', encoding='utf-8') as jsonl_file:
            jsonl_file.write(json.dumps(entry) + '\n')

    def save_question_img_and_answer(self, G, path_count, filename, jsonl_path):
        """Save the graph images and return the answer choices"""
        self.generate_graph_image(G, filename, False)  # False for undirected
        
        # Generate options
        correct, offsets = self.generate_options(G, path_count)
        if correct is None or offsets is None:
            return None, None
        
        # Combine all options and shuffle
        option_list = [correct] + offsets
        random.shuffle(option_list)
        
        # Find index of correct answer after shuffling
        correct_idx = option_list.index(correct)
        
        # Get the adjacency matrix
        matrix = nx.adjacency_matrix(G).todense().tolist()
        
        # Update the JSONL file
        self.update_jsonl(matrix, option_list, self.source_node, self.target_node, correct_idx, jsonl_path)
        
        return correct, offsets

    def generate_graph_image(self, graph, filename, is_directed):
        """Generate and save an image of the graph"""
        generator = GraphImageGenerator(graph, is_directed, filename)
        generator.generate_and_display()


def main(generation_cnt=15, benchmark_root='dataset'):

    TASK_NAME = 'path_counting'
    task_count = 0
    
    # Remove existing JSONL file if it exists
    jsonl_path = os.path.join(benchmark_root, f"{TASK_NAME}.jsonl")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    img_dir = os.path.join(benchmark_root, TASK_NAME)
    # make dir if it doesn't exist
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    
    for task_count in trange(generation_cnt, desc="Generating Tasks", unit="task"):
        # Step 1: Generate graph
        task = PathCountingTask()
        G, path_count = task.generate_random_graph()
        
        if not G or path_count is None:
            print("Failed to generate suitable graph, retrying...")
            continue
            
        # Step 2: Save the task
        img_file_path = os.path.join(benchmark_root, TASK_NAME, f'{task_count}.png')
        correct, offsets = task.save_question_img_and_answer(G, path_count, img_file_path, jsonl_path)
        
        if correct is None or offsets is None:
            print("Failed to generate valid options, retrying...")
            continue

if __name__ == "__main__":
    main(generation_cnt=500)