import os
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from graph_tasks.config import GRAPH_SIZE, NODE_SIZE, EDGE_COLOR, FONT_SIZE, TEXT_COLOR, NODE_COLOR, HIGHLIGHT_COLOR, ARROW_SIZE, NODE_LIMIT, EDGE_LIMIT
import hashlib

def hash_str(filename):
    return hashlib.md5(filename.encode()).hexdigest()

class GraphImageGenerator:
    def __init__(self, G, task_is_directed, filename, source=None, target=None):
        self.G = G
        self.filename = filename
        self.source = source
        self.target = target
        self.is_directed = isinstance(G, nx.DiGraph)
        
        # Generate positions in a circle for an n-sided polygon
        self.pos = self.circular_layout(G)
        
    def circular_layout(self, G):
        # Get the number of nodes in the graph
        n = len(G.nodes())
        
        # Angle between each node (in radians)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        
        # Calculate the positions of each node
        pos = {}
        for i, node in enumerate(G.nodes()):
            x = np.cos(angles[i])
            y = np.sin(angles[i])
            pos[node] = (x, y)
            return pos
        
    def draw_graph(self, node_color='white', edge_color='black'):
        # Make sure all nodes have positions
        if self.pos is None or not all(node in self.pos for node in self.G.nodes()):
            # If positions are missing, generate positions for all nodes
            self.pos = nx.circular_layout(self.G)
            
        # Use white for all nodes and add black edge color for node contours
        node_colors = ['white' for node in self.G.nodes()]
        
        if self.is_directed:
            nx.draw(self.G, self.pos, with_labels=True, 
                    node_size=NODE_SIZE, 
                    node_color=node_colors,
                    edgecolors='black',  # Add black contour to nodes
                    font_size=FONT_SIZE, 
                    font_weight='bold',
                    edge_color=edge_color,
                    arrowsize=ARROW_SIZE,
                    width=2.0)
        else:
            nx.draw(self.G, self.pos, with_labels=True, 
                    node_size=NODE_SIZE, 
                    node_color=node_colors,
                    edgecolors='black',  # Add black contour to nodes
                    font_size=FONT_SIZE, 
                    font_weight='bold',
                    edge_color=edge_color)

        
    # def draw_graph(self, node_color, edge_color):
    #     node_colors = [self.source_color(node) if node == self.source else self.target_color(node) if node == self.target else node_color for node in self.G.nodes()]
        
    #     if self.is_directed:
    #         nx.draw(self.G, self.pos, with_labels=True, node_size=NODE_SIZE, node_color=node_colors, 
    #                 font_size=FONT_SIZE, font_weight='bold', edge_color=edge_color,
    #                 arrowsize=ARROW_SIZE, width=2.0)
    #     else:
    #         nx.draw(self.G, self.pos, with_labels=True, node_size=NODE_SIZE, node_color=node_colors, 
    #                 font_size=FONT_SIZE, font_weight='bold', edge_color=edge_color)

    def source_color(self, node):
        return 'red' if node == self.source else NODE_COLOR

    def target_color(self, node):
        return 'red' if node == self.target else NODE_COLOR

    def display_weights(self):
        if nx.get_edge_attributes(self.G, 'weight'):
            edge_labels = nx.get_edge_attributes(self.G, 'weight')
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=10)

    def generate_and_display(self, node_color=NODE_COLOR, edge_color=EDGE_COLOR, show=False):
        plt.figure(figsize=GRAPH_SIZE)
        self.draw_graph(node_color, edge_color)
        self.display_weights()
        plt.tight_layout()
        plt.savefig(self.filename, format="PNG", dpi=80)
        if show:
            plt.show()


class RandomGraphTask:
    def __init__(self, task_type):
        self.task_type = task_type
        self.task_is_directed = task_type in ['cycle_detection']
        self.dir_graph = None
        self.undir_graph = None
        self.graph = None
        self.correct_choice = []
        self.tricky_choice = []
        self.incorrect_choices = []

    def filter_graph(self, graph):
        raise NotImplementedError("This method must be implemented in a subclass")

    def generate_random_graph(self):
        raise NotImplementedError("This method must be implemented in a subclass")
    
    def generate_correct_answer(self):
        raise NotImplementedError("This method must be implemented in a subclass")
    
    def generate_tricky_answer(self):
        raise NotImplementedError("This method must be implemented in a subclass")

    def generate_incorrect_answers(self):
        raise NotImplementedError("This method must be implemented in a subclass")
    
    def generate_answers(self):
        self.correct_choice = self.generate_correct_answer()
        self.tricky_choice = self.generate_tricky_answer()
        self.incorrect_choices = self.generate_incorrect_answers()
        return self.correct_choice, self.tricky_choice, self.incorrect_choices
        
    def save_question_img_and_answer(self):
        raise NotImplementedError("This method must be implemented in a subclass")
    
    def generate_graph_image(self, graph, filename, node_color=NODE_COLOR, edge_color=EDGE_COLOR):
        # Check if task is cycle detection, and use specialized image generator if needed
        if self.task_type == "cycle_detection":
            generator = GraphImageGenerator(graph, self.task_is_directed, filename)
        else:
            generator = GraphImageGenerator(graph, filename)
        generator.generate_and_display(node_color, edge_color)
        
    def graph_to_text_representation(self, graph):
        # Convert graph to a text representation (Edge List, Adjacency Matrix, or Adjacency List)
        edge_list = list(graph.edges())
        adj_matrix = nx.to_numpy_array(graph)  # Use `to_numpy_array` instead of `to_numpy_matrix`
        adj_list = {node: list(graph.neighbors(node)) for node in graph.nodes()}
        return edge_list, adj_matrix, adj_list
