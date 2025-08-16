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
from collections import defaultdict, deque
from graph_tasks.base_tasks import GraphImageGenerator
from graph_tasks.config import *

class BFSTraversalOrderTask:
    def __init__(self):
        self.start_node = None
        self.correct_order = None
        self.level_groups = None  # Store nodes by their BFS level
        self.task_type = "bfs_traversal"

    def generate_random_graph(self):
        max_attempts = 50
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            # Step 1: Randomly generate an undirected graph G
            num_nodes = random.randint(NODE_LIMITS['bfs_traversal'][0], NODE_LIMITS['bfs_traversal'][1])
            # Adjust edge probability to get interesting BFS traversals
            # Too high: too many nodes at same level, too low: linear paths
            edge_probability = min(1.8 / num_nodes, 0.25)  
            G = nx.gnp_random_graph(num_nodes, edge_probability, directed=False)
            
            # Check if the graph is connected
            if not nx.is_connected(G):
                continue
                
            # Choose a random starting node
            nodes = list(G.nodes())
            if len(nodes) < 5:  # Need at least a few nodes for interesting BFS
                continue
                
            # Try different starting nodes to find one with interesting BFS traversal
            random.shuffle(nodes)
            
            for start_node in nodes:
                self.start_node = start_node
                
                # Compute BFS traversal order and group by levels
                level_groups = self.compute_bfs_levels(G, start_node)
                
                # We want at least 3 levels for interesting questions
                if len(level_groups) < 3:
                    continue
                    
                # We want levels with different numbers of nodes
                # (makes it easier to distinguish different traversal orders)
                level_sizes = [len(level) for level in level_groups]
                if len(set(level_sizes[1:])) <= 1:  # All non-root levels same size
                    continue
                    
                # We want reasonable numbers of nodes at each level
                # Not too many (confusing) or too few (trivial)
                if any(len(level) > 5 for level in level_groups):
                    continue
                    
                # Store the level groups for this traversal
                self.level_groups = level_groups
                
                # Compute the flattened correct BFS order
                self.correct_order = []
                for level in level_groups:
                    self.correct_order.extend(sorted(level))  # Sort within each level
                
                return G
            
        print(f"Failed to generate suitable graph after {max_attempts} attempts")
        return None

    def compute_bfs_levels(self, G, start_node):
        """
        Compute BFS traversal order grouped by levels.
        Returns a list of lists, where each inner list contains nodes at the same level.
        """
        visited = set([start_node])
        queue = deque([(start_node, 0)])  # (node, level)
        level_groups = [[start_node]]  # First level contains only the start node
        
        while queue:
            node, level = queue.popleft()
            
            # If we're about to process a node at a new level, create a new level group
            if level + 1 >= len(level_groups):
                level_groups.append([])
                
            # Add neighbors to the queue, record their level
            for neighbor in sorted(G.neighbors(node)):  # Sort for deterministic BFS
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, level + 1))
                    level_groups[level + 1].append(neighbor)
        
        # Remove any empty levels at the end (shouldn't happen with connected graphs)
        while level_groups and not level_groups[-1]:
            level_groups.pop()
            
        return level_groups

    def generate_confusing_distractor(self):
        """
        Generate a confusing distractor by swapping nodes between different depths.
        E.g., if correct is [1; 2,3; 4,5], distractor might be [1; 2,4; 3,5]
        """
        if not self.level_groups or len(self.level_groups) < 3:
            return None
            
        # Copy the level groups to avoid modifying the original
        distractor_levels = [list(level) for level in self.level_groups]
        
        # Try to swap nodes between different (non-root) levels
        # We leave the root level (level 0) unchanged for clarity
        swap_attempts = 0
        max_swap_attempts = 10
        
        while swap_attempts < max_swap_attempts:
            swap_attempts += 1
            
            # Choose two different non-root levels
            eligible_levels = list(range(1, len(distractor_levels)))
            if len(eligible_levels) < 2:
                continue
                
            level1_idx, level2_idx = random.sample(eligible_levels, 2)
            level1, level2 = distractor_levels[level1_idx], distractor_levels[level2_idx]
            
            # If either level is empty, try again
            if not level1 or not level2:
                continue
                
            # Choose a random node from each level to swap
            node1 = random.choice(level1)
            node2 = random.choice(level2)
            
            # Swap the nodes
            level1.remove(node1)
            level2.remove(node2)
            level1.append(node2)
            level2.append(node1)
            
            # If there are more than 2 levels, try to make one more swap
            if len(eligible_levels) > 2 and random.random() < 0.7:  # 70% chance for second swap
                # Choose two more levels (could include previously swapped levels)
                level1_idx, level2_idx = random.sample(eligible_levels, 2)
                level1, level2 = distractor_levels[level1_idx], distractor_levels[level2_idx]
                
                if level1 and level2:  # If both levels have nodes
                    node1 = random.choice(level1)
                    node2 = random.choice(level2)
                    
                    # Swap the nodes
                    level1.remove(node1)
                    level2.remove(node2)
                    level1.append(node2)
                    level2.append(node1)
            
            # Flatten the levels to get the traversal order
            distractor_order = []
            for level in distractor_levels:
                distractor_order.extend(sorted(level))  # Sort within each level
                
            # Check if the distractor is different from the correct order
            if distractor_order != self.correct_order:
                return distractor_order
        
        # If all swap attempts failed, return None
        return None

    def generate_incorrect_options(self):
        """
        Generate completely incorrect options by reshuffling the levels.
        E.g., if correct is [1; 2,3; 4,5], incorrect might be [1; 4,5; 2,3]
        """
        if not self.level_groups or len(self.level_groups) < 3:
            return []
            
        incorrect_options = []
        max_attempts = 15
        attempts = 0
        
        while len(incorrect_options) < 2 and attempts < max_attempts:
            attempts += 1
            
            # Copy the level groups to avoid modifying the original
            shuffled_levels = [list(level) for level in self.level_groups]
            
            # Keep the root level (level 0) fixed, shuffle the rest
            non_root_levels = shuffled_levels[1:]
            random.shuffle(non_root_levels)
            
            # Reconstruct with root level fixed
            shuffled_levels = [shuffled_levels[0]] + non_root_levels
            
            # Flatten the levels to get the traversal order
            incorrect_order = []
            for level in shuffled_levels:
                incorrect_order.extend(sorted(level))  # Sort within each level
                
            # Check if this option is different from the correct answer and any existing options
            if (incorrect_order != self.correct_order and 
                not any(self.are_orders_equal(incorrect_order, option) for option in incorrect_options)):
                incorrect_options.append(incorrect_order)
        
        # If we couldn't generate 2 distinct incorrect options, use more aggressive reshuffling
        while len(incorrect_options) < 2 and attempts < max_attempts * 2:
            attempts += 1
            
            # Completely shuffle all nodes across all levels (except root)
            all_non_root_nodes = []
            for level in self.level_groups[1:]:
                all_non_root_nodes.extend(level)
                
            random.shuffle(all_non_root_nodes)
            
            # Distribute shuffled nodes into the same level structure
            shuffled_levels = [list(self.level_groups[0])]  # Keep root level
            node_index = 0
            
            for level in self.level_groups[1:]:
                level_size = len(level)
                shuffled_levels.append(all_non_root_nodes[node_index:node_index + level_size])
                node_index += level_size
            
            # Flatten to get traversal order
            incorrect_order = []
            for level in shuffled_levels:
                incorrect_order.extend(sorted(level))  # Sort within each level
                
            # Check if this option is different from the correct answer and any existing options
            if (incorrect_order != self.correct_order and 
                not any(self.are_orders_equal(incorrect_order, option) for option in incorrect_options)):
                incorrect_options.append(incorrect_order)
        
        return incorrect_options[:2]  # Return up to 2 options

    def are_orders_equal(self, order1, order2):
        """Check if two traversal orders are equivalent"""
        return order1 == order2

    def format_traversal_order(self, order):
        """
        Format a traversal order for display in the question.
        Group nodes by their level, separated by semicolons.
        """
        if not self.level_groups or not order:
            return "Invalid order"
            
        # Reconstruct the level structure from the flattened order
        formatted_levels = []
        node_index = 0
        
        for level in self.level_groups:
            level_size = len(level)
            level_nodes = order[node_index:node_index + level_size]
            formatted_levels.append(",".join(map(str, sorted(level_nodes))))
            node_index += level_size
            
        return ",".join(formatted_levels)
        # return "; ".join(formatted_levels)

    def update_jsonl(self, matrix, options, start_node, correct_idx, level_sizes, jsonl_path):
        """
        Update a JSONL file with new data about BFS traversal order.
        
        Args:
            matrix (list): The adjacency matrix of the undirected graph.
            options (list): The BFS traversal options (already formatted).
            start_node (int): The starting node for BFS.
            correct_idx (int): The index of the correct traversal.
            level_sizes (list): The number of nodes at each BFS level.
        """
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        
        # Prepare the data entry
        entry = {
            "matrix": matrix,
            "options": options,
            "start_node": start_node,
            "correct_idx": correct_idx,
            "level_sizes": level_sizes
        }
        
        # Append the new entry to the JSONL file
        with open(jsonl_path, 'a', encoding='utf-8') as jsonl_file:
            jsonl_file.write(json.dumps(entry) + '\n')

    def save_question_img_and_answer(self, G, filename, jsonl_path):
        """Save the graph image and return the answer choices"""
        self.generate_graph_image(G, filename, False)  # False for undirected
        
        # Generate options
        correct_order = self.correct_order
        confusing_distractor = self.generate_confusing_distractor()
        incorrect_options = self.generate_incorrect_options()
        
        if not correct_order or not confusing_distractor or len(incorrect_options) < 2:
            return None
        
        # Format all options (group nodes by level)
        formatted_correct = self.format_traversal_order(correct_order)
        formatted_distractor = self.format_traversal_order(confusing_distractor)
        formatted_incorrect1 = self.format_traversal_order(incorrect_options[0])
        formatted_incorrect2 = self.format_traversal_order(incorrect_options[1])
        
        # Combine all formatted options and shuffle
        option_list = [formatted_correct, formatted_distractor, 
                      formatted_incorrect1, formatted_incorrect2]
        random.shuffle(option_list)
        
        # Find index of correct answer after shuffling
        correct_idx = option_list.index(formatted_correct)
        
        # Get the adjacency matrix
        matrix = nx.adjacency_matrix(G).todense().tolist()
        
        # Get the level sizes for reference
        level_sizes = [len(level) for level in self.level_groups]
        
        # Update the JSONL file
        self.update_jsonl(matrix, option_list, self.start_node, correct_idx, level_sizes, jsonl_path)
        
        return option_list, correct_idx

    def generate_graph_image(self, graph, filename, is_directed):
        """Generate and save an image of the graph"""
        generator = GraphImageGenerator(graph, is_directed, filename)
        generator.generate_and_display()


def main(generation_cnt=15, benchmark_root='dataset'):
    TASK_NAME = 'bfs_traversal'
    task_count = 0

    jsonl_path = os.path.join(benchmark_root, f"{TASK_NAME}.jsonl")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)


    img_dir = os.path.join(benchmark_root, TASK_NAME)
    # make dir if it doesn't exist
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    
    for task_count in trange(generation_cnt, desc="Generating Tasks", unit="task"):
        # Step 1: Generate graph
        task = BFSTraversalOrderTask()
        G = task.generate_random_graph()
        
        if not G:
            print("Failed to generate suitable graph, retrying...")
            continue
            
        # Step 2: Save the task
        img_file_path = os.path.join(benchmark_root, TASK_NAME, f'{task_count}.png')
           
        options, correct_idx = task.save_question_img_and_answer(G, img_file_path, jsonl_path)
        
        if not options:
            print("Failed to generate valid options, retrying...")
            continue

if __name__ == "__main__":
    main(generation_cnt=500)