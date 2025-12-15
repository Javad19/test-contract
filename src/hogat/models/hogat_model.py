# Core HOGAT model implementation. I built this around the multi-hop attention idea to grab those long-range deps better than standard GAT.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, softmax

class HOGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, num_heads=8, dropout=0.3, k_hops=2):
        """
        Higher-Order Graph Attention Network. Tuned with 8 heads since it balanced perf in our tests.
        
        Args:
            input_dim: Input feature size from the graph.
            hidden_dim: Hidden units—64 worked well without overfitting.
            output_dim: 5 for the vuln types we target.
            num_layers: Stuck with 2 to keep it lightweight.
            num_heads: Multi-head attention count.
            dropout: 0.3 dropout to prevent overfit on ESC data.
            k_hops: Max hops; 2 is optimal from ablations.
        """
        super(HOGAT, self).__init__()
        self.num_layers = num_layers
        self.k_hops = k_hops
        self.dropout = dropout
        self.convs = nn.ModuleList()
        
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        
        for _ in range(1, num_layers):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        
        self.output = nn.Linear(hidden_dim * num_heads, output_dim)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        
        # For gating in multi-hop (rough sim of Eq 8)
        self.gate_mlp = nn.Linear(hidden_dim * num_heads * (k_hops + 1), hidden_dim * num_heads)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        adj = to_dense_adj(edge_index).squeeze(0)  # For powering
        
        for i in range(self.num_layers):
            hop_embs = [x]  # Start with 0-hop (self)
            
            # Compute k-hop embeddings with powering for better range capture
            current_adj = adj.clone()
            for hop in range(1, self.k_hops + 1):
                x_hop = self.convs[i](hop_embs[-1], edge_index)
                # Attention scores with powering (approx Eq 5-7)
                alpha = softmax(self.leaky_relu(x_hop), dim=-1)
                current_adj = torch.mm(current_adj, adj)  # Power for higher order
                hop_embs.append(alpha * x_hop)
            
            # Gate and fuse hops (inspired by Eq 8-9)
            fused = torch.cat(hop_embs, dim=-1)
            x = self.gate_mlp(fused)  # Simple MLP gate
            if i < self.num_layers - 1:
                x = self.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.output(x)
        x = self.sigmoid(x)
        
        # Quick debug during dev
        # print(f"Output shape: {x.shape}")
        return x
