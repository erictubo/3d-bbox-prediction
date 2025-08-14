import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
import json

from src.models.model import ThreeDBBoxModel
from src.dataset import BBoxDataset, flatten_collate
from src.config import *
from src.utils.losses import HybridLoss


# Function to build data_paths from root (unchanged)
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

def main():
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
    criterion = HybridLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Resume logic
    start_epoch = 0
    if os.path.exists('model_checkpoint.pth'):
        checkpoint = torch.load('model_checkpoint.pth')
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
        
        for batch_idx, batch in enumerate(train_loader):
            rgb_crops = batch['rgb_crops'].to(DEVICE)
            points = batch['points'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)  # (B, 10)

            preds = model(rgb_crops, points)  # (B, 10)
            loss = criterion(preds, targets)
            loss.backward()
            
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
            if batch_idx % ACCUM_STEPS != 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)} - Train Loss: {loss.item():.4f}")

            acc_train_loss += loss.item()

            if (batch_idx + 1) % ACCUM_STEPS == 0:
                loss /= ACCUM_STEPS
                optimizer.step()
                optimizer.zero_grad()

            # TODO: log specific losses
            # corner loss
            # iou loss
            # angular loss

            # TODO: visualize predictions

            # TODO: evaluate on test set

            # TODO: review losses

            # TODO: documentation

            if batch_idx % ACCUM_STEPS == 0:
                avg_train_loss = acc_train_loss / len(train_loader)
                acc_train_loss /= ACCUM_STEPS

                # Accumulated validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        rgb_crops = batch['rgb_crops'].to(DEVICE)
                        points = batch['points'].to(DEVICE)
                        targets = batch['targets'].to(DEVICE)
                        preds = model(rgb_crops, points)
                        val_loss += criterion(preds, targets).item()
                    acc_val_loss = val_loss / len(val_loader)
                    writer.add_scalar('Loss/val_batch', acc_val_loss, epoch * len(train_loader) + batch_idx)
                    print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)} - Train Loss: {acc_train_loss:.4f} - Val Loss: {acc_val_loss:.4f}")
                model.train()

                acc_train_loss = 0.0

                # TODO: scheduler step before epoch end

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                rgb_crops = batch['rgb_crops'].to(DEVICE)
                points = batch['points'].to(DEVICE)
                targets = batch['targets'].to(DEVICE)
                preds = model(rgb_crops, points)
                val_loss += criterion(preds, targets).item()
            avg_val_loss = val_loss / len(val_loader)

        # Log to TensorBoard
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Avg Train Loss: {avg_train_loss:.4f} - Avg Val Loss: {avg_val_loss:.4f}")

        # Checkpointing: Save every epoch + best val
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, 'model_checkpoint.pth')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'model_best.pth')
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
    main()