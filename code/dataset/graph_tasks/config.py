# config.py
GRAPH_SIZE = (5, 5)        # Size of the plot
NODE_SIZE = 3000           # Size of the nodes
EDGE_COLOR = 'black'       # Color of the edges
FONT_SIZE = 16             # Font size for the labels
TEXT_COLOR = 'black'       # Color of the text
NODE_COLOR = 'white'       # Changed default node color to white
# Node contour is set directly in draw_graph function
HIGHLIGHT_COLOR = 'yellow' # Highlight color for specific nodes or edges
ARROW_SIZE = 20            # Size of the arrows for directed graphs

# Difficulty settings for the random graph generation
NODE_LIMIT = (4, 8)   # Random number of nodes will be between NODE_LIMIT[0] and NODE_LIMIT[1]
NODE_LIMITS = {
    'path_existence': (6, 9),
    'path_counting': (6, 9),
    'shortest_path': (6, 9),
    'bfs_traversal': (6, 9),
}
EDGE_LIMIT = (1, 20)   # Random number of edges will be between EDGE_LIMIT[0] and EDGE_LIMIT[1]
