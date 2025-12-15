
# Builds the contract graph. Nodes for vulns, funcs, vars; edges from control/data flows. Used Slither for the heavy lifting.

import torch
from torch_geometric.data import Data, HeteroData
from slither.slither import Slither
import networkx as nx
import numpy as np
from .utils import compute_centrality
import tempfile
import os
from torch.nn.functional import one_hot

# Keywords pulled from Table 1—expanded a bit based on common patterns
VULN_KEYWORDS = {
    'reentrancy': ['call', 'send', 'transfer', 'msg.sender', 'msg.value', 'delegatecall', 'selfdestruct'],
    'timestamp': ['block.timestamp', 'now', 'block.number'],
    'overflow': ['+', '-', '*', '/'],
    'infinite_loop': ['while', 'for', 'do-while']
}

def construct_graph(normalized_code: str) -> Data:
    """
    Puts together the Contract Graph with heterogeneous nodes and multi-rel edges.
    """
    with tempfile.NamedTemporaryFile(suffix='.sol', delete=False) as temp_file:
        temp_file.write(normalized_code.encode())
        temp_path = temp_file.name
    
    slither = Slither(temp_path)
    os.unlink(temp_path)
    
    vuln_nodes = []  # V1: indicators
    func_nodes = []  # V2: functions
    var_nodes = []   # V3: variables
    
    node_types = []  # For one-hot
    keyword_feats = []  # Basic embedding
    
    for contract in slither.contracts:
        for stmt in contract.functions + contract.modifiers:
            for expr in stmt.expressions:
                for kw_type, kws in VULN_KEYWORDS.items():
                    for kw in kws:
                        if kw in str(expr):
                            vuln_nodes.append(kw)
                            node_types.append(0)  # Type 0 for vuln
                            keyword_feats.append(kw_type)  # String for now; encode later
        
        func_nodes.extend([f.name for f in contract.functions_declared])
        for f in func_nodes:
            node_types.append(1)  # Type 1 for func
        
        var_nodes.extend([v.name for v in contract.variables if not v.is_constant])
        for v in var_nodes:
            node_types.append(2)  # Type 2 for var
    
    num_nodes = len(vuln_nodes) + len(func_nodes) + len(var_nodes)
    if num_nodes == 0:
        return Data(x=torch.empty((0, 128)), edge_index=torch.empty((2, 0), dtype=torch.long))
    
    # Features: one-hot type (3 dims) + dummy keyword vec (5 dims for categories) + centrality (1 dim) padded to 128
    type_onehot = one_hot(torch.tensor(node_types), num_classes=3).float()
    kw_vec = torch.zeros((num_nodes, len(VULN_KEYWORDS)))  # One per category
    for idx, kw in enumerate(keyword_feats):
        if kw in VULN_KEYWORDS:
            kw_idx = list(VULN_KEYWORDS.keys()).index(kw)
            kw_vec[idx, kw_idx] = 1.0
    features = torch.cat([type_onehot, kw_vec, torch.zeros(num_nodes, 128 - 3 - len(VULN_KEYWORDS))], dim=1)
    
    # Edges: Approximate control/data from Slither
    edge_index = []
    edge_types = []  # 0: control, 1: data, 2: fallback
    for contract in slither.contracts:
        for func in contract.functions:
            # Control edges from CFG
            cfg = func.control_flow_graph
            for node in cfg.nodes:
                for child in node.sons:
                    edge_index.append([node.id, child.id])  # Assume node ids map
                    edge_types.append(0)
            # Data edges from vars
            for var in func.variables_read_or_written:
                # Connect func to var (dummy index mapping)
                func_idx = func_nodes.index(func.name) if func.name in func_nodes else -1
                var_idx = var_nodes.index(var.name) + len(func_nodes) if var.name in var_nodes else -1
                if func_idx != -1 and var_idx != -1:
                    edge_index.append([func_idx, var_idx])
                    edge_types.append(1)
    
    if edge_index:
        edge_index = torch.tensor(edge_index).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Centrality on nx graph
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(edge_index.t().tolist())
    centrality = compute_centrality(nx_graph)
    centrality_tensor = torch.tensor(centrality[:num_nodes]).unsqueeze(1).float()  # Pad if needed
    features[:, -1] = centrality_tensor.squeeze()  
    graph = Data(x=features, edge_index=edge_index)
   
    return graph