# Loads and processes datasets into PyG format. Assumes label files alongside .sol (e.g., esc_labels.json from dataset sources).

import os
import torch
from torch_geometric.data import InMemoryDataset
from hogat.tools.normalization import normalize_code
from hogat.tools.graph_construction import construct_graph
import json  # For labels

class SmartContractDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw', self.name)

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed', self.name)

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.sol')]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for raw_file in self.raw_file_names:
            raw_path = os.path.join(self.raw_dir, raw_file)
            with open(raw_path, 'r') as f:
                code = f.read()
            normalized = normalize_code(code)
            graph = construct_graph(normalized)
            
            # Load labels: Assume per-dataset JSON like {'file.sol': [vuln_types binary array]}
            label_path = os.path.join(self.raw_dir, f'{self.name}_labels.json')
            if os.path.exists(label_path):
                with open(label_path, 'r') as lf:
                    labels = json.load(lf)
                graph_y = torch.tensor(labels.get(raw_file, [0]*5), dtype=torch.float).unsqueeze(0).repeat(graph.num_nodes, 1)
            else:
                graph_y = torch.zeros((graph.num_nodes, 5))  # Default non-vuln; warn in log
                # print(f"No labels for {raw_file}, defaulting to zeros.")
            graph.y = graph_y
            
            data_list.append(graph)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def load_dataset(dataset_name, root='data', split_ratio=0.8, seed=42):
    torch.manual_seed(seed)  # For repro splits
    dataset = SmartContractDataset(root=root, name=dataset_name)
    num_train = int(len(dataset) * split_ratio)
    indices = torch.randperm(len(dataset))
    train_data = dataset[indices[:num_train]]
    test_data = dataset[indices[num_train:]]
    return train_data, test_data
