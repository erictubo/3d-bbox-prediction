"""
Training Script

Handles data loading and model training with validation.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
import json

from src.model import ThreeDBBoxModel
from src.dataset import BBoxDataset, flatten_collate
from src.config import *
from src.utils.losses import HybridLoss


def build_data_paths(root_dir):
    data_paths = []
    for subdir, _, files in os.walk(root_dir):
        if 'rgb.jpg' in files and 'mask.npy' in files and 'pc.npy' in files and 'bbox3d.npy' in files:
            data_paths.append({
                'image': os.path.join(subdir, 'rgb.jpg'),
                'mask': os.path.join(subdir, 'mask.npy'),
                'pc': os.path.join(subdir, 'pc.npy'),
                'bbox': os.path.join(subdir, 'bbox3d.npy')
            })
    return data_paths

def train_model(model_name='model', loss_weights=dict[str, float]):
    """
    Train the model.
    - model_name: str, name of the model
        -> loads 'models/{model_name}_checkpoint.pth' if available
        -> saves 'models/{model_name}_best.pth'
    - loss_weights: dict[str, float], weights for each loss term
        -> center: weight for center loss
        -> diag: weight for diagonal loss
        -> quat: weight for quaternion loss
        -> canonical: weight for canonical angular loss
        -> corner: weight for corner distance loss
        -> iou: weight for projected IoU loss
    """
    # Load or generate splits (unchanged)
    if os.path.exists('splits.json'):
        with open('splits.json', 'r') as f:
            splits = json.load(f)
        train_paths = splits['train']
        val_paths = splits['val']
    else:
        all_paths = build_data_paths(DATA_ROOT)
        train_paths, temp_paths = train_test_split(all_paths, train_size=TRAIN_SPLIT, random_state=42)
        val_paths, test_paths = train_test_split(temp_paths, train_size=VAL_SPLIT / (1 - TRAIN_SPLIT), random_state=42)
        splits = {'train': train_paths, 'val': val_paths, 'test': test_paths}
        with open('splits.json', 'w') as f:
            json.dump(splits, f)

    # Datasets (unchanged)
    train_dataset = BBoxDataset(train_paths, max_points=MAX_POINTS, crop_size=CROP_SIZE)
    val_dataset = BBoxDataset(val_paths, max_points=MAX_POINTS, crop_size=CROP_SIZE)

    # Loaders with small batch size for testing
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=flatten_collate, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=flatten_collate, num_workers=NUM_WORKERS)
    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
    print(f"Train loader length: {len(train_loader)}, Val loader length: {len(val_loader)}")

    # Model setup (unchanged)
    model = ThreeDBBoxModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    criterion = HybridLoss(loss_weights)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Resume logic
    start_epoch = 0
    if os.path.exists(f'models/{model_name}_checkpoint.pth'):
        checkpoint = torch.load(f'models/{model_name}_checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # TensorBoard writer and early stopping vars
    writer = SummaryWriter('runs/test_training')  # Logs to ./runs
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop with reduced epochs for testing
    num_epochs = NUM_EPOCHS
    for epoch in range(start_epoch, num_epochs):
        model.train()
        avg_train_loss = 0.0
        acc_train_loss = 0.0
        avg_train_loss_dict = {'center': 0.0, 'diag': 0.0, 'quat': 0.0, 'canonical': 0.0, 'corner': 0.0, 'iou': 0.0}
        acc_train_loss_dict = {'center': 0.0, 'diag': 0.0, 'quat': 0.0, 'canonical': 0.0, 'corner': 0.0, 'iou': 0.0}
        
        for batch_idx, batch in enumerate(train_loader):
            rgb_crops = batch['rgb_crops'].to(DEVICE)
            points = batch['points'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)  # (B, 10)

            preds = model(rgb_crops, points)  # (B, 10)
            loss, loss_dict = criterion(preds, targets)
            loss.backward()
            
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
            for key, value in loss_dict.items():
                if value is not None:
                    writer.add_scalar(f'Loss/train_batch_{key}', value, epoch * len(train_loader) + batch_idx)
                    acc_train_loss_dict[key] += value / ACCUM_STEPS
                    avg_train_loss_dict[key] += value / len(train_loader)

            if batch_idx % ACCUM_STEPS != 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)} - Train Loss: {loss.item():.4f}")
                print(f"- Losses: {loss_dict}")

            acc_train_loss += loss.item()

            if (batch_idx + 1) % ACCUM_STEPS == 0:
                loss /= ACCUM_STEPS
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % ACCUM_STEPS == 0:
                avg_train_loss = acc_train_loss / len(train_loader)
                acc_train_loss /= ACCUM_STEPS

                # Accumulated validation
                model.eval()
                acc_val_loss = 0.0
                acc_val_loss_dict = {'center': 0.0, 'diag': 0.0, 'quat': 0.0, 'canonical': 0.0, 'corner': 0.0, 'iou': 0.0}

                with torch.no_grad():
                    for batch in val_loader:
                        rgb_crops = batch['rgb_crops'].to(DEVICE)
                        points = batch['points'].to(DEVICE)
                        targets = batch['targets'].to(DEVICE)
                        preds = model(rgb_crops, points)
                        val_loss, val_loss_dict = criterion(preds, targets)
                        acc_val_loss += val_loss.item() / len(val_loader)

                        for key, value in val_loss_dict.items():
                            if value is not None:
                                acc_val_loss_dict[key] += value / len(val_loader)

                    writer.add_scalar('Loss/val', acc_val_loss, epoch * len(train_loader) + batch_idx)
                    for key, value in acc_val_loss_dict.items():
                        if value is not None:
                            writer.add_scalar(f'Loss/val_{key}', value, epoch * len(train_loader) + batch_idx)
                    print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)} - Train Loss: {acc_train_loss:.4f} - Val Loss: {acc_val_loss:.4f}")
                    print(f"- Losses: {acc_val_loss_dict}")
                model.train()

                acc_train_loss = 0.0

                # TODO: scheduler step before epoch end

        # Validation
        model.eval()
        avg_val_loss = 0.0
        avg_val_loss_dict = {'center': 0.0, 'diag': 0.0, 'quat': 0.0, 'canonical': 0.0, 'corner': 0.0, 'iou': 0.0}

        with torch.no_grad():
            for batch in val_loader:
                rgb_crops = batch['rgb_crops'].to(DEVICE)
                points = batch['points'].to(DEVICE)
                targets = batch['targets'].to(DEVICE)
                preds = model(rgb_crops, points)
                val_loss, val_loss_dict = criterion(preds, targets)
                avg_val_loss += val_loss.item() / len(val_loader)

                for key, value in val_loss_dict.items():
                    if value is not None:
                        avg_val_loss_dict[key] += value / len(val_loader)
                
            

        # Log to TensorBoard
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        for key, value in avg_val_loss_dict.items():
            if value is not None:
                writer.add_scalar(f'Loss/val_epoch_{key}', value, epoch)

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Avg Train Loss: {avg_train_loss:.4f} - Avg Val Loss: {avg_val_loss:.4f}")

        # Checkpointing: Save every epoch + best val
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, f'models/{model_name}_checkpoint.pth')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'models/{model_name}_best.pth')
            print("Saved best model")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    writer.close()  # Close TensorBoard writer

if __name__ == '__main__':
    # train_model(model_name='model', loss_weights={'center': 0.4, 'diag': 0.3, 'quat': 0.3, 'canonical': 0.0, 'corner': 0.0, 'iou': 0.0})
    pass