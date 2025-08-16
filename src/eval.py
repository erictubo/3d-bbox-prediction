"""
Evaluation Script

Handles model evaluation with visualization on the test set.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
from torch.utils.data import DataLoader
import json

from src.model import ThreeDBBoxModel
from src.dataset import BBoxDataset, flatten_collate
from src.utils.visualization import visualize_results_batch
from src.utils.losses import HybridLoss
from src.config import *


def evaluate_model(model_name='model', loss_weights=dict[str, float], visualization=True):
    """
    Evaluate the model on the test set.
    - model_name: str, name of the model 
            -> loads models/{model_name}_best.pth
    - visualization: bool (default: True)
    """
    # Load splits
    with open('splits.json', 'r') as f:
        splits = json.load(f)
    test_paths = splits['test']

    # Dataset/Loader
    test_dataset = BBoxDataset(test_paths, max_points=MAX_POINTS, crop_size=CROP_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=flatten_collate, num_workers=NUM_WORKERS)

    # Load model
    model = ThreeDBBoxModel().to(DEVICE)
    model.load_state_dict(torch.load(f'models/{model_name}.pth'))
    model.eval()

    # Metrics accumulators
    total_loss = 0.0
    total_loss_dict = {'center': 0.0, 'diag': 0.0, 'quat': 0.0, 'canonical': 0.0, 'corner': 0.0, 'iou': 0.0}

    criterion = HybridLoss(loss_weights)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            rgb_crops = batch['rgb_crops'].to(DEVICE)
            points = batch['points'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            preds = model(rgb_crops, points)
            
            loss, loss_dict = criterion(preds, targets)
            total_loss += loss.item() * preds.size(0)
            for key, value in loss_dict.items():
                if value is not None:
                    total_loss_dict[key] += value / len(test_loader)

            if visualization:
                visualize_results_batch(preds, targets, points, batch_idx, save_dir=f'vis/eval/{model_name}')

    avg_loss = total_loss / len(test_loader)
    avg_loss_dict = {key: value for key, value in total_loss_dict.items()}

    print(f"Avg Test Loss: {avg_loss:.4f}")
    print(f"Avg Losses: {avg_loss_dict}")


if __name__ == '__main__':
    evaluate_model(model_name='model 2', visualization=True)
