
# Training script for HOGAT—main loop with some early stopping logic.

import argparse
import yaml
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from hogat.models import HOGAT
from hogat.tools.utils import setup_logging, evaluate_model
from hogat.tools.data_utils import load_dataset

logger = setup_logging()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/hogat_default.yaml')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    torch.manual_seed(config['data']['seed'])
    
    # Data—loop over datasets if needed, but default to ESC
    train_data, val_data = load_dataset(config['datasets']['esc'], split_ratio=0.7)  # 70/30 for train/val
    _, test_data = load_dataset(config['datasets']['esc'], split_ratio=0.8)  # Full test
    train_loader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['training']['batch_size'])
    test_loader = DataLoader(test_data, batch_size=config['training']['batch_size'])
    
    model = HOGAT(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        k_hops=config['model']['k_hops']
    )
    
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'], eps=config['training']['optimizer_params']['eps'])
    scheduler = ReduceLROnPlateau(optimizer, patience=5)
    criterion = torch.nn.BCELoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_val_loss = float('inf')
    patience = 10 if config['training']['early_stopping'] else config['training']['epochs']
    counter = 0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")
        
        val_metrics = evaluate_model(model, val_loader, device)
        val_loss = 1 - val_metrics['f1']  # Proxy loss from F1
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'checkpoints/hogat_best.pth')
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping triggered.")
                break
    
    # Load best and test
    model.load_state_dict(torch.load('checkpoints/hogat_best.pth'))
    test_metrics = evaluate_model(model, test_loader, device)
    logger.info(f"Final Test Metrics: {test_metrics}")

if __name__ == "__main__":
    main()
