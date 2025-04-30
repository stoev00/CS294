import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_scatter import scatter
from torch_geometric.nn import radius_graph

# Define edges as a list of (source, target) node pairs
edges = torch.tensor([
    [0, 1],  # Edge from node 0 -> 1
    [1, 2],  # Edge from node 1 -> 2
    [2, 0],  # Edge from node 2 -> 0
    [2, 3]   # Edge from node 2 -> 3
], dtype=torch.long)

num_nodes = edges.max().item() + 1  # Find the number of nodes dynamically

# Extract source and target nodes
source_nodes = edges[:, 0]
target_nodes = edges[:, 1]

# Create indices for the adjacency matrix
indices = source_nodes * num_nodes + target_nodes  # Flattened indices for a 1D tensor

# Initialize adjacency matrix as a flat tensor
adj_matrix = torch.zeros(num_nodes * num_nodes, dtype=torch.float32)

# Scatter ones into the adjacency matrix
scatter(torch.ones_like(indices, dtype=torch.float32), indices, out=adj_matrix, reduce="sum")

# Reshape back to 2D adjacency matrix
adj_matrix = adj_matrix.view(num_nodes, num_nodes)

print(adj_matrix)