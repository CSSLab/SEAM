import random, networkx as nx, matplotlib.pyplot as plt, numpy as np
from graph_tasks.config import *
import time, tqdm
from tqdm import trange
import json
import os
from collections import defaultdict
from graph_tasks.base_tasks import GraphImageGenerator

class PathExistenceTask:
    def __init__(self):
        self.source_node = None
        self.target_node = None
        self.correct_choice = None
        self.tricky_choice = None
        self.incorrect_choices = []
        self.task_type = "path_existence"

    def generate_random_graph(self):
        max_attempts = 50
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            # Step 1: Randomly generate a directed graph D
            num_nodes = random.randint(NODE_LIMITS['path_existence'][0], NODE_LIMITS['path_existence'][1])
            # Increase edge probability to get more interesting paths
            edge_probability = min(2.0 / num_nodes, 0.3)  # Ensure reasonable density
            G = nx.gnp_random_graph(num_nodes, edge_probability, directed=True)
            D = nx.DiGraph(G)  # Ensure it's directed
            
            # Convert D to undirected graph
            G_undirected = D.to_undirected()
            
            # Check if the graph is connected
            if not nx.is_weakly_connected(D):
                continue
                
            # Choose random source and target nodes
            nodes = list(D.nodes())
            if len(nodes) < 2:
                continue
                
            # Try different source-target pairs to find one that works
            source_target_pairs = []
            for src in nodes:
                for tgt in nodes:
                    if src != tgt:
                        source_target_pairs.append((src, tgt))
                        
            random.shuffle(source_target_pairs)
            
            found_suitable_pair = False
            for src, tgt in source_target_pairs:
                self.source_node = src
                self.target_node = tgt
                
                # Check if path exists in D from src to tgt
                if not nx.has_path(D, src, tgt):
                    continue
                    
                # Get all paths in both graph types
                try:
                    # Limit path generation to prevent excessive computation
                    directed_paths = list(nx.all_simple_paths(D, src, tgt, cutoff=6))
                    undirected_paths = list(nx.all_simple_paths(G_undirected, src, tgt, cutoff=6))
                    
                    # If no paths found or too many paths (computationally expensive), skip
                    if not directed_paths or len(directed_paths) > 100 or len(undirected_paths) > 100:
                        continue
                        
                    # Check if there's at least one path in undirected that's not in directed
                    has_different_paths = False
                    for u_path in undirected_paths:
                        if not any(self.is_same_path(u_path, d_path) for d_path in directed_paths):
                            has_different_paths = True
                            break
                            
                    if has_different_paths:
                        found_suitable_pair = True
                        break
                except nx.NetworkXError:
                    # Handle any NetworkX errors (e.g., excessive path computation)
                    continue
            
            if found_suitable_pair:
                return D, G_undirected
                
        print(f"Failed to generate suitable graph after {max_attempts} attempts")
        return None, None

    def is_same_path(self, path1, path2):
        """Check if two paths are the same (same sequence of nodes)"""
        if len(path1) != len(path2):
            return False
            
        for i in range(len(path1)):
            if path1[i] != path2[i]:
                return False
                
        return True

    def generate_correct_answer(self, D):
        """
        Generate the correct choice - a valid path that exists in the directed graph D.
        Must start with source_node and end with target_node.
        """
        if not nx.has_path(D, self.source_node, self.target_node):
            return None
            
        try:
            # Get all simple paths from source to target in the directed graph
            # Use cutoff to prevent excessive computation
            all_paths = list(nx.all_simple_paths(D, self.source_node, self.target_node, cutoff=6))
            if not all_paths:
                return None
                
            # Choose a random path as the correct answer
            return random.choice(all_paths)
        except nx.NetworkXError:
            # Handle potential computation errors
            return None

    def generate_tricky_answer(self, D, G):
        """
        Generate the tricky choice - a path that exists in undirected G but NOT in directed D.
        Must start with source_node and end with target_node.
        """
        try:
            # Use cutoff to prevent excessive computation
            directed_paths = list(nx.all_simple_paths(D, self.source_node, self.target_node, cutoff=6))
            undirected_paths = list(nx.all_simple_paths(G, self.source_node, self.target_node, cutoff=6))
            
            # Find paths in G that are not in D
            tricky_paths = []
            for u_path in undirected_paths:
                # Verify it starts with source and ends with target
                if u_path[0] == self.source_node and u_path[-1] == self.target_node:
                    # Check it's not in any of the directed paths
                    if not any(self.is_same_path(u_path, d_path) for d_path in directed_paths):
                        tricky_paths.append(u_path)
            
            if tricky_paths:
                return random.choice(tricky_paths)
            return None
        except nx.NetworkXError:
            # Handle potential computation errors
            return None

    def generate_non_path(self, D, G):
        """
        Generate a sequence of nodes from source to target that is NOT a valid path in either graph.
        IMPORTANT: All options must start with source node and end with target node.
        """
        nodes = list(G.nodes())
        max_attempts = 50
        
        # Get all nodes except source and target
        middle_nodes = [n for n in nodes if n != self.source_node and n != self.target_node]
        
        # Strategy 1: Create a path with invalid edges (nodes not connected)
        for _ in range(max_attempts):
            # Always start with source node
            path = [self.source_node]
            current = self.source_node
            
            # Add 1-3 middle nodes (or fewer if not enough nodes available)
            middle_count = min(random.randint(1, 3), len(middle_nodes))
            if middle_count > 0:
                # Try to select nodes that break the path
                for _ in range(middle_count):
                    # Prefer nodes that are NOT neighbors of current node to break the path
                    non_neighbors = [n for n in middle_nodes if n not in list(G.neighbors(current)) and n not in path]
                    
                    # If no non-neighbors available, use any remaining middle node
                    available_nodes = non_neighbors if non_neighbors else [n for n in middle_nodes if n not in path]
                    
                    if not available_nodes:
                        break
                        
                    next_node = random.choice(available_nodes)
                    path.append(next_node)
                    current = next_node
            
            # Always end with target node
            path.append(self.target_node)
            
            # Verify this isn't accidentally a valid path
            if not self.is_valid_path(D, path) and not self.is_valid_path(G, path):
                return path
                
        # Strategy 2: Create a path with a cycle or repeated node (invalid simple path)
        for _ in range(max_attempts):
            # Start with source
            path = [self.source_node]
            
            # Add middle nodes (we need at least one to create a repetition)
            if middle_nodes:
                # Get a random middle node
                middle = random.choice(middle_nodes)
                # Add it twice to create a non-simple path
                path.append(middle)
                path.append(middle)
                
                # Optionally add one more random middle node
                if len(middle_nodes) > 1 and random.choice([True, False]):
                    another = random.choice([n for n in middle_nodes if n != middle])
                    path.append(another)
            
            # End with target
            path.append(self.target_node)
            
            # This should never be a valid simple path due to the repetition
            if not self.is_valid_path(D, path) and not self.is_valid_path(G, path):
                return path
        
        # Strategy 3: Just use random nodes but ensure source start and target end
        for _ in range(max_attempts):
            # Start with source
            path = [self.source_node]
            
            # Add random middle nodes (1-3)
            middle_sample_size = min(random.randint(1, 3), len(middle_nodes))
            if middle_sample_size > 0:
                middle_sample = random.sample(middle_nodes, middle_sample_size)
                path.extend(middle_sample)
            
            # End with target
            path.append(self.target_node)
            
            # Check if this randomly happens to be a valid path
            if not self.is_valid_path(D, path) and not self.is_valid_path(G, path):
                return path
                
        # Fallback: If all else fails, create a minimal invalid path
        # Simply insert a node that's guaranteed to break the path if one exists
        all_edges = list(G.edges())
        if all_edges and middle_nodes:
            return [self.source_node, middle_nodes[0], self.target_node]
        else:
            # Last resort: Just source and target with something in between
            middle_value = random.choice(nodes) if nodes else 999
            return [self.source_node, middle_value, self.target_node]

    def is_valid_path(self, graph, path):
        """Check if a sequence of nodes forms a valid path in the graph"""
        if len(path) < 2:
            return True  # A single node is trivially a valid path
            
        for i in range(len(path) - 1):
            if not graph.has_edge(path[i], path[i + 1]):
                return False
                
        return True

    def are_paths_equivalent(self, path1, path2):
        """Check if two paths are equivalent"""
        return self.is_same_path(path1, path2)

    def generate_incorrect_answers(self, D, G):
        """Generate incorrect answers that are not paths from source to target in either graph"""
        incorrect_answers = []
        
        max_attempts = 50
        attempts = 0
        
        while len(incorrect_answers) < 2 and attempts < max_attempts:
            attempts += 1
            
            # Generate a non-path
            path = self.generate_non_path(D, G)
            
            if not path:
                continue
                
            # Ensure it's not a duplicate of an already chosen incorrect answer
            is_duplicate = False
            for existing_path in incorrect_answers:
                if self.are_paths_equivalent(path, existing_path):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                incorrect_answers.append(path)
                
        # If we can't generate two distinct incorrect answers, return None
        if len(incorrect_answers) < 2:
            print("Failed to generate two distinct incorrect answers")
            return None
            
        return incorrect_answers

    def generate_answers(self, D, G):
        """Generate all answer choices and validate them"""
        self.correct_choice = self.generate_correct_answer(D)
        if not self.correct_choice:
            return None, None, None
            
        self.tricky_choice = self.generate_tricky_answer(D, G)
        if not self.tricky_choice:
            return None, None, None
            
        self.incorrect_choices = self.generate_incorrect_answers(D, G)
        if not self.incorrect_choices or len(self.incorrect_choices) < 2:
            return None, None, None
            
        return self.correct_choice, self.tricky_choice, self.incorrect_choices
    
    def update_jsonl(self, matrix, options, source_node, target_node, correct_idx, tricky_idx, jsonl_path):
        """
        Update a JSONL file with new data about path existence.
        
        Args:
            matrix (list): The adjacency matrix of the directed graph.
            options (list): The path options.
            source_node (int): The source node.
            target_node (int): The target node.
            correct_idx (int): The index of the correct path.
            tricky_idx (int): The index of the tricky path.
        """
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        
        # Prepare the data entry
        entry = {
            "matrix": matrix,
            "options": options,
            "source_node": source_node,
            "target_node": target_node,
            "correct_idx": correct_idx,
            "tricky_idx": tricky_idx
        }
        
        # Append the new entry to the JSONL file
        with open(jsonl_path, 'a', encoding='utf-8') as jsonl_file:
            jsonl_file.write(json.dumps(entry) + '\n')

    def save_question_img_and_answer(self, D, G, filename, jsonl_path):
        """Save the graph images and return the answer choices"""
        self.generate_graph_image(D, filename, True)  # True for directed
        # jsonl format:
        # matrix, options, source_node, target_node, correct_idx, tricky_idx

        option_list = [self.correct_choice, self.tricky_choice] + self.incorrect_choices

        # shuffle the options, keep in mind the correct answer index
        random.shuffle(option_list)

        correct_idx = option_list.index(self.correct_choice)
        tricky_idx = option_list.index(self.tricky_choice)

        # D: DiGraph, we want the adjacency matrix of it
        matrix = nx.adjacency_matrix(D).todense().tolist()

        self.update_jsonl(matrix, option_list, self.source_node, self.target_node, correct_idx, tricky_idx, jsonl_path)

        return self.correct_choice, self.tricky_choice, self.incorrect_choices

    def generate_graph_image(self, graph, filename, is_directed):
        """Generate and save an image of the graph"""
        generator = GraphImageGenerator(graph, is_directed, filename)
        generator.generate_and_display()


def main(generation_cnt=15, benchmark_root='dataset'):
    task_count = 0
    TASK_NAME = 'path_existence'

    jsonl_path = os.path.join(benchmark_root, f"{TASK_NAME}.jsonl")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

        
    img_dir = os.path.join(benchmark_root, TASK_NAME)
    # make dir if it doesn't exist
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
   
     
    for task_count in trange(generation_cnt, desc="Generating Tasks", unit="task"):
        # Step 1: Generate graphs
        task = PathExistenceTask()
        D, G = task.generate_random_graph()
        
        if not D or not G:
            print("Failed to generate suitable graphs, retrying...")
            continue
            
        # Step 2: Generate answers
        correct, tricky, incorrect = task.generate_answers(D, G)
        
        if not correct or not tricky or not incorrect:
            print("Failed to generate valid answer choices, retrying...")
            continue
            
        img_file_path = os.path.join(benchmark_root, TASK_NAME, f'{task_count}.png')
        task.save_question_img_and_answer(D, G, img_file_path, jsonl_path)

if __name__ == "__main__":
    main(generation_cnt=500)