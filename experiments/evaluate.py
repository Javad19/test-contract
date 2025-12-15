# experiments/evaluate.py
# Eval script for models on specific datasets. Handy for RQ checks.

import argparse
import yaml
import torch
from torch_geometric.loader import DataLoader
from hogat.models import HOGAT
from hogat.tools.utils import evaluate_model
from hogat.tools.data_utils import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/hogat_default.yaml')
    parser.add_argument('--model_path', type=str, default='checkpoints/hogat_best.pth')  # Use best from train
    parser.add_argument('--dataset', type=str, default='esc')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model = HOGAT(**config['model'])  # Shorthand init
    model.load_state_dict(torch.load(args.model_path))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    _, test_data = load_dataset(args.dataset)
    test_loader = DataLoader(test_data, batch_size=config['training']['batch_size'])
    
    metrics = evaluate_model(model, test_loader, device)
    print(f"Eval on {args.dataset}: {metrics}")
    
    with open(f'results/{args.dataset}_eval.yaml', 'w') as f:
        yaml.dump(metrics, f)

if __name__ == "__main__":
    main()
