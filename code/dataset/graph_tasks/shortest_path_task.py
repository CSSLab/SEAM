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
from graph_tasks.base_tasks import GraphImageGenerator
from graph_tasks.config import *

class ShortestPathTask:
    def __init__(self):
        self.source_node = None
        self.target_node = None
        self.correct_answer = None
        self.offset_answers = []
        self.task_type = "shortest_path"

    def generate_random_graph(self):
        max_attempts = 50
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            # Step 1: Randomly generate an undirected graph G
            num_nodes = random.randint(NODE_LIMITS['shortest_path'][0], NODE_LIMITS['shortest_path'][1])
            edge_probability = min(2.0 / num_nodes, 0.3)  # Ensure reasonable density
            G = nx.gnp_random_graph(num_nodes, edge_probability, directed=False)
            
            # Check if the graph is connected
            if not nx.is_connected(G):
                continue
                
            # Choose random source and target nodes
            nodes = list(G.nodes())
            if len(nodes) < 2:
                continue
                
            # Try different source-target pairs to find one that works
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
                    
                # Find shortest path length
                shortest_path_length = nx.shortest_path_length(G, src, tgt)
                
                # Ensure path is non-trivial (not just adjacent nodes)
                if shortest_path_length >= 2:
                    # Also verify that the shortest path is unique or close to unique
                    # to avoid ambiguity
                    try:
                        all_paths = list(nx.all_simple_paths(G, src, tgt, cutoff=shortest_path_length + 1))
                        # Count paths of exactly the shortest length
                        shortest_paths = [p for p in all_paths if len(p) - 1 == shortest_path_length]
                        
                        # We want questions where the shortest path is reasonably clear
                        # Too many shortest paths makes the question ambiguous
                        if len(shortest_paths) <= 3:
                            return G, shortest_path_length
                    except nx.NetworkXError:
                        # Handle any NetworkX errors
                        continue
            
        print(f"Failed to generate suitable graph after {max_attempts} attempts")
        return None, None

    def generate_options(self, G, shortest_length):
        """
        Generate the options for the shortest path task.
        Returns the correct answer and a list of offset answers.
        """
        if shortest_length is None:
            return None, None
        
        # Calculate edge count for offset calculation
        edge_count = G.number_of_edges()
        offset = math.ceil(0.1 * edge_count + 1)
        
        # Correct answer is the shortest path length
        self.correct_answer = shortest_length
        
        # Generate offset options (correct answer ± offset)
        offset_options = []
        
        # Add offset answers, ensuring they're positive
        for delta in range(-offset, offset + 1):
            if delta != 0:  # Skip the correct answer
                candidate = shortest_length + delta
                if candidate > 0:  # Avoid negative or zero values
                    offset_options.append(candidate)
        
        # If we don't have enough options, add more by increasing the offset
        while len(offset_options) < 3:
            offset += 1
            candidate_plus = shortest_length + offset
            candidate_minus = shortest_length - offset
            
            if candidate_minus > 0 and candidate_minus not in offset_options:
                offset_options.append(candidate_minus)
                
            if len(offset_options) < 3 and candidate_plus not in offset_options:
                offset_options.append(candidate_plus)
        
        # Ensure we have exactly 3 offset options
        if len(offset_options) > 3:
            offset_options = sorted(offset_options)[:3]
        
        self.offset_answers = offset_options
        return self.correct_answer, self.offset_answers

    def update_jsonl(self, matrix, options, source_node, target_node, correct_idx, jsonl_path):
        """
        Update a JSONL file with new data about shortest path.
        
        Args:
            matrix (list): The adjacency matrix of the undirected graph.
            options (list): The path length options.
            source_node (int): The source node.
            target_node (int): The target node.
            correct_idx (int): The index of the correct length.
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

    def save_question_img_and_answer(self, G, shortest_length, filename, jsonl_path):
        """Save the graph images and return the answer choices"""
        self.generate_graph_image(G, filename, False)  # False for undirected
        
        # Generate options
        correct, offsets = self.generate_options(G, shortest_length)
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
    task_count = 0
    TASK_NAME = 'shortest_path'

    jsonl_path = os.path.join(benchmark_root, f"{TASK_NAME}.jsonl")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

        
    img_dir = os.path.join(benchmark_root, TASK_NAME)
    # make dir if it doesn't exist
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
  
    
    for task_count in trange(generation_cnt, desc="Generating Tasks", unit="task"):
        # Step 1: Generate graph
        task = ShortestPathTask()
        G, shortest_length = task.generate_random_graph()
        
        if not G or shortest_length is None:
            print("Failed to generate suitable graph, retrying...")
            continue
            
        # Step 2: Save the task
        img_file_path = os.path.join(benchmark_root, TASK_NAME, f'{task_count}.png')
        correct, offsets = task.save_question_img_and_answer(G, shortest_length, img_file_path, jsonl_path)
    
        # if correct is None or offsets is None:
        #     print("Failed to generate valid options, retrying...")
        #     continue

if __name__ == "__main__":
    main(generation_cnt=500)