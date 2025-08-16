"""
Visualization Utilities

This module provides visualization functions for 3D bounding box datasets.
It handles raw data (image + masks + point cloud + bboxes), processed data (rgb crop + points + bbox),
and results (pred vs gt on points).

Key Functions:
- visualize_raw_data: Visualize raw data from files
- visualize_raw_data_path: Visualize raw data from path
- visualize_processed_data: Visualize processed data from dataloader
- visualize_processed_data_batch: Visualize full batch of processed data from dataloader
- visualize_results: Visualize prediction vs ground truth on points
- visualize_results_batch: Visualize full batch of predictions vs ground truth on points
- visualize_loss_breakdown: Visualize loss breakdown as a bar chart
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.utils.bbox_conversion import corners_from_params_torch


def _visualize_raw_data_objects(image, mask, pc, bbox, n=None, save_dir='vis_raw', name='raw'):
    """
    Core visualization function that works with objects directly.
    - image: numpy array (H, W, 3) or PIL Image
    - mask: numpy array (N, H, W)
    - pc: numpy array (3, H, W) or (N, 3) for point cloud data
    - bbox: numpy array (N, 8, 3)
    - n: int, specific object (or None for all)
    - save_dir: str, folder to save plots
    - name: str -> {name}_image_masks.png, {name}_pc_bboxes.png
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert PIL Image to numpy if needed
    if hasattr(image, 'convert'):  # PIL Image
        image = np.array(image)
    
    # Convert torch tensors to numpy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(pc, torch.Tensor):
        pc = pc.cpu().numpy()
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.cpu().numpy()

    N = mask.shape[0]
    if n is None:
        n = N  # Show all
    
    # Image + Masks
    fig, ax = plt.subplots()
    ax.imshow(image)
    if n < N:
        ax.set_title(f"Raw Image with Mask {n}")
        masked_data = np.ma.masked_where(~mask[n], mask[n])
        ax.imshow(masked_data, alpha=0.8, cmap="Reds")
    else:
        ax.set_title(f"Raw Image with All {N} Masks")
        cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
        alphas = [1.0] * 6 + [0.8] * 6 + [0.6] * 6 + [0.4] * 6
        for i in range(N):
            masked_data = np.ma.masked_where(~mask[i], mask[i])
            ax.imshow(masked_data, alpha=alphas[i % len(alphas)], cmap=cmaps[i % len(cmaps)])
    plt.savefig(os.path.join(save_dir, f'{name}_image_masks.png'))
    plt.close()

    # Point Cloud + BBoxes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Handle different point cloud formats
    if pc.ndim == 3 and pc.shape[0] == 3:  # (3, H, W) format
        # Reshape point cloud from (3, H, W) to (N, 3) where N = H*W
        pc_reshaped = pc.reshape(3, -1).T  # (N, 3) where N = H*W
    else:  # (N, 3) format
        pc_reshaped = pc
    
    ax.scatter(pc_reshaped[:, 0], pc_reshaped[:, 1], pc_reshaped[:, 2], s=1, c='b', alpha=0.1, label='Points')
    if n < N:
        ax.set_title(f"Raw Point Cloud with BBox {n}")
        ax.scatter(bbox[n, :, 0], bbox[n, :, 1], bbox[n, :, 2], s=10, c='r', label='BBox')
    else:
        ax.set_title(f"Raw Point Cloud with All {N} BBoxes")
        colors = ['grey', 'purple', 'blue', 'green', 'orange', 'red']
        alphas = [1.0] * 6 + [0.8] * 6 + [0.6] * 6 + [0.4] * 6
        for i in range(N):
            ax.scatter(bbox[i, :, 0], bbox[i, :, 1], bbox[i, :, 2], s=10, c=colors[i % len(colors)], alpha=alphas[i % len(alphas)], label=f'BBox {i}')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(os.path.join(save_dir, f'{name}_pc_bboxes.png'))
    plt.close()


def visualize_raw_data(image_path, mask_path, pc_path, bbox_path, n=None, save_dir='vis_raw', name='raw'):
    """
    Visualize raw data from files.
    - image_path: str to rgb.jpg
    - mask_path: str to mask.npy (N, H, W)
    - pc_path: str to pc.npy (3, H, W)
    - bbox_path: str to bbox3d.npy (N, 8, 3)
    - n: int, specific object (or None for all)
    - save_dir: str, folder to save plots
    - name: str -> {name}_image_masks.png, {name}_pc_bboxes.png
    """
    # Load data from files
    image = plt.imread(image_path)  # (H, W, 3)
    mask = np.load(mask_path)  # (N, H, W)
    pc = np.load(pc_path)  # (3, H, W)
    bbox = np.load(bbox_path)  # (N, 8, 3)
    
    # Call the core visualization function
    _visualize_raw_data_objects(image, mask, pc, bbox, n=n, save_dir=save_dir, name=name)


def visualize_raw_data_objects(image, mask, points, bbox_corners, save_dir='vis_raw', name='raw'):
    """
    Visualize raw data from objects (PIL Image, numpy arrays, tensors).
    - image: PIL Image object
    - mask: numpy array (H, W) or (N, H, W)
    - points: torch tensor (N, 3) or numpy array
    - bbox_corners: numpy array (N, 8, 3) or (8, 3)
    - save_dir: str, folder to save plots
    - name: str -> {name}_image_masks.png, {name}_pc_bboxes.png
    """
    # Handle mask shape
    if mask.ndim == 2:
        mask = mask[np.newaxis, :, :]  # (1, H, W)
    
    # Handle bbox_corners shape
    if bbox_corners.ndim == 2:
        bbox_corners = bbox_corners[np.newaxis, :, :]  # (1, 8, 3)
    
    # Call the core visualization function
    _visualize_raw_data_objects(image, mask, points, bbox_corners, save_dir=save_dir, name=name)

def visualize_raw_data_path(path, save_dir='vis_raw', name='raw'):
    """
    Visualize raw data from path.
    - path: str to folder containing rgb.jpg, mask.npy, pc.npy, bbox3d.npy
    - save_dir: str, folder to save plots
    - name: str -> {name}_image_masks.png, {name}_pc_bboxes.png
    """
    visualize_raw_data(os.path.join(path, 'rgb.jpg'), os.path.join(path, 'mask.npy'),
        os.path.join(path, 'pc.npy'), os.path.join(path, 'bbox3d.npy'), save_dir=save_dir, name=name)


def visualize_processed_data(rgb_crops, points_list, targets, idx, save_dir='vis_processed', name='processed'):
    """
    Visualize processed data from dataloader.
    - rgb_crops: (B, 3, 224, 224) tensor
    - points_list: (B, max_points, 3) tensor
    - targets: (B, 10) tensor (params)
    - idx: int, batch index to visualize
    - save_dir: str
    - name: str -> {name}_{idx}_rgb.png, {name}_{idx}_points_bbox.png
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # RGB Crop (first in batch)
    crop = rgb_crops[idx].permute(1, 2, 0).cpu().numpy()  # (224, 224, 3)
    plt.imshow(crop)
    plt.title("Processed RGB Crop")
    plt.savefig(os.path.join(save_dir, f'{name}_{idx}_rgb.png'))
    plt.close()
    
    # Points + BBox from Targets
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = points_list[idx].cpu().numpy()  # (max_points, 3)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='b', label='Processed Points')
    
    gt_corners = corners_from_params_torch(targets[idx, :3], targets[idx, 3:6], targets[idx, 6:]).cpu().numpy()  # (8, 3)
    edges = [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]]  # Connect bbox edges
    for edge in edges:
        ax.plot(gt_corners[edge, 0], gt_corners[edge, 1], gt_corners[edge, 2], 'r-')
    ax.set_title("Processed Points with GT BBox")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(os.path.join(save_dir, f'{name}_{idx}_points_bbox.png'))
    plt.close()

def visualize_processed_data_batch(rgb_crops, points_list, targets, batch_idx, save_dir='vis_processed', name=None):
    """
    Visualize full batch of processed data from dataloader.
    - rgb_crops: (B, 3, 224, 224) tensor
    - points_list: (B, max_points, 3) tensor
    - targets: (B, 10) tensor (params)
    - batch_idx: int, batch index
    - save_dir: str
    """
    if name is None: name = f'batch_{batch_idx}'
    os.makedirs(save_dir, exist_ok=True)
    for i in range(rgb_crops.size(0)):
        visualize_processed_data(rgb_crops, points_list, targets, i, save_dir=save_dir, name=name)


def visualize_results(preds, targets, points, idx, save_dir='vis_results', name='results'):
    """
    Visualize prediction vs ground truth on points.
    - preds: (B, 10) tensor
    - targets: (B, 10) tensor
    - points: (B, max_points, 3) tensor
    - idx: int, batch index
    - save_dir: str
    - name: str -> {name}_{idx}.png
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points_np = points[idx].cpu().numpy()
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=1, c='b', label='Points')
    
    # GT BBox (red)
    gt_corners = corners_from_params_torch(targets[idx, :3], targets[idx, 3:6], targets[idx, 6:]).cpu().numpy()
    edges = [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]]
    for edge in edges:
        ax.plot(gt_corners[edge, 0], gt_corners[edge, 1], gt_corners[edge, 2], 'r-', label='GT' if edge == edges[0] else "")
    
    # Pred BBox (green)
    pred_corners = corners_from_params_torch(preds[idx, :3], preds[idx, 3:6], preds[idx, 6:]).cpu().numpy()
    for edge in edges:
        ax.plot(pred_corners[edge, 0], pred_corners[edge, 1], pred_corners[edge, 2], 'g-', label='Pred' if edge == edges else "")
    
    ax.set_title(f"Pred vs GT BBox on Points (Batch {idx})")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(os.path.join(save_dir, f'{name}_{idx}.png'))
    plt.close()

def visualize_results_batch(preds, targets, points, batch_idx, save_dir='vis_results'):
    """
    Visualize full batch of predictions vs ground truth on points.
    - preds: (B, 10) tensor
    - targets: (B, 10) tensor
    - points: (B, max_points, 3) tensor
    - batch_idx: int, batch index
    - save_dir: str
    """
    os.makedirs(save_dir, exist_ok=True)
    for i in range(preds.size(0)):
        visualize_results(preds, targets, points, i, save_dir=save_dir, name=f'batch_{batch_idx}')

def visualize_loss_breakdown(losses: torch.Tensor, metric_name: str = 'Corner Distance Loss', save_dir: str = 'vis_loss', batch_idx: int = 0):
    """
    Plot per-sample loss breakdown as a bar chart.
    - losses: (B,) tensor of per-sample losses (e.g., from corner_distance_loss)
    - metric_name: str for title
    - save_dir: str
    - batch_idx: int for filename
    """
    os.makedirs(save_dir, exist_ok=True)
    B = losses.size(0)
    sample_indices = list(range(B))
    loss_values = losses.cpu().numpy()
    
    plt.figure(figsize=(8, 4))
    plt.bar(sample_indices, loss_values, color='skyblue')
    plt.xlabel('Sample Index')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Breakdown per Sample (Batch {batch_idx})')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'loss_breakdown_batch_{batch_idx}.png'))
    plt.close()