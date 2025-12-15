# Misc helpers: logging, metrics, centrality. Kept metrics macro avg for class imbalance.

import logging
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import networkx as nx

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true_np = y_true.cpu().numpy()
    y_pred_np = (y_pred > 0.5).cpu().numpy()
    
    metrics = {
        'accuracy': accuracy_score(y_true_np, y_pred_np, average='macro'),
        'precision': precision_score(y_true_np, y_pred_np, average='macro', zero_division=0),
        'recall': recall_score(y_true_np, y_pred_np, average='macro', zero_division=0),
        'f1': f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    }
    return metrics

def compute_centrality(graph: nx.Graph) -> np.ndarray:
    if len(graph) == 0:
        return np.zeros(1)
    try:
        centrality = nx.eigenvector_centrality(graph, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        centrality = nx.degree_centrality(graph)  # Fallback if convergence fails
    values = np.array(list(centrality.values()))
    normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
    return normalized

def evaluate_model(model, loader, device):
    """Central eval func—used across scripts to avoid copy-paste."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            y_pred.append(out)
            y_true.append(batch.y)
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    return compute_metrics(y_true, y_pred)
