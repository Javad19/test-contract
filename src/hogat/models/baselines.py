# src/hogat/models/baselines.py
# Baseline models for RQ3: GraphSAGE, Generic GNN, GCN, GAT.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, MessagePassing

class GraphSAGE(nn.Module):
    """GraphSAGE baseline"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        for _ in range(1, num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.output(x)
        return torch.sigmoid(x)

class GenericGNN(MessagePassing):
    """Generic GNN baseline. Simple message passing."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GenericGNN, self).__init__(aggr='add')
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.num_layers = num_layers

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for _ in range(self.num_layers):
            x = self.propagate(edge_index, x=x)
            x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return torch.sigmoid(x)

    def message(self, x_j):
        return x_j

class GCN(nn.Module):
    """Graph Convolutional Network baseline."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        for _ in range(1, num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.output(x)
        return torch.sigmoid(x)

class GAT(nn.Module):
    """Graph Attention Network baseline."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, num_layers=2):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList([GATConv(input_dim, hidden_dim, heads=num_heads)])
        for _ in range(1, num_layers):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads))
        self.output = nn.Linear(hidden_dim * num_heads, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        x = self.output(x)
        return torch.sigmoid(x)