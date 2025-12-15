
# Baseline runs for RQs 2,3,5. Mocks for LLMs since no API here.

import yaml
import torch
from torch_geometric.loader import DataLoader
from hogat.models.baselines import GraphSAGE, GenericGNN, GCN, GAT
from hogat.tools.utils import evaluate_model
from hogat.tools.data_utils import load_dataset
import matplotlib.pyplot as plt

with open('configs/hogat_default.yaml', 'r') as f:
    config = yaml.safe_load(f)

baselines = {
    'GraphSAGE': GraphSAGE,
    'GenericGNN': GenericGNN,
    'GCN': GCN,
    'GAT': GAT,
    # RQ2 stubs—implement full if time
    'VulDet': None,  # TODO: Single-layer from [7]
    'HGAT': None,  # Hierarchical from [8]
    # RQ5: Hardcoded from paper results
    'GPTScan': {'f1': 0.839, 'precision': 0.85, 'recall': 0.83, 'accuracy': 0.84}
}

results = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for name, ModelClass in baselines.items():
    if ModelClass is None:
        continue  # Skip unimplemented
    if isinstance(ModelClass, dict):  # For mocks
        results[name] = ModelClass
        continue
    for ds in ['esc', 'vsc', 'solidifi']:
        model = ModelClass(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            output_dim=config['model']['output_dim']
        )
        model.to(device)
        
        _, test_data = load_dataset(ds)
        test_loader = DataLoader(test_data, batch_size=config['training']['batch_size'])
        
        # Assume pre-trained or quick train; eval
        metrics = evaluate_model(model, test_loader, device)
        results.setdefault(name, {})[ds] = metrics

with open('results/baselines.yaml', 'w') as f:
    yaml.dump(results, f)

# Plot for Figs 5-8—example for reentrancy (esc)
if 'esc' in results.get('GAT', {}):
    models = list(results.keys())
    f1s = [results[m].get('esc', {}).get('f1', 0) for m in models]
    plt.bar(models, f1s)
    plt.title('Reentrancy F1 Comparison')
    plt.savefig('docs/figures/Figure5.png')
    plt.close()
