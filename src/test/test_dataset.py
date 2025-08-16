"""
Testing of Dataset Module
- dataset.py
- bbox_conversion.py

1. BBOX CONVERSION
2. RAW DATA VISUALIZATION
3. DATASET CREATION
4. PROCESSED DATA VISUALIZATION
5. DATALOADER
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import torch
import random
import numpy as np
from torch.utils.data import DataLoader

from src.dataset import BBoxDataset, flatten_collate
from src.config import *
from src.utils.bbox_conversion import params_from_corners_quat, corners_from_params_numpy, corners_from_params_torch
from src.utils.visualization import visualize_raw_data_path, visualize_processed_data_batch


def test_bbox_conversion_numpy(corners_np, tolerance=1e-2, mode='full'):  # **Added mode param for flexibility**
    """
    Test params_from_corners and corners_from_params round-trip.
    mode='full' (default): Corners -> params -> corners with mean min dist.
    mode='quat': Quat -> corners -> quat (assumes input is corners, extracts quat).
    """
    params = params_from_corners_quat(corners_np)
    center = torch.from_numpy(params[:3]).float()
    dims = torch.from_numpy(params[3:6]).float()
    quat = torch.from_numpy(params[6:10]).float()
    
    if mode == 'quat':
        # Quat -> corners -> quat
        corners = corners_from_params_numpy(center, dims, quat)
        new_params = params_from_corners_quat(corners)  # Back to params
        new_quat = new_params[6:10]
        dist = np.linalg.norm(quat - new_quat)  # Quaternion distance (norm diff, or use angular if needed)
        if dist <= tolerance:
            print(f"Quat round-trip successful! Dist: {dist}")
            return True
        else:
            print(f"Quat round-trip failed! Dist: {dist}")
            return False
    
    # Full mode (corners -> params -> corners)
    reconstructed = corners_from_params_numpy(center, dims, quat)
    
    # Compute min distance per point (accounts for permutations)
    from scipy.spatial.distance import cdist
    dists = cdist(corners_np, reconstructed)
    min_dists = np.min(dists, axis=1)
    mean_dist = np.mean(min_dists)
    
    if mean_dist <= tolerance:
        print(f"Round-trip successful! Mean dist: {mean_dist}")
        return True
    else:
        print(f"Round-trip failed! Mean dist: {mean_dist}")
        return False
    
def test_bbox_conversion_torch(corners_np, tolerance=1e-2, mode='full'):
    """
    Test params_from_corners_quat and corners_from_params_quat_torch round-trip, handling quat equivalence.
    """
    params = params_from_corners_quat(corners_np)
    center = torch.from_numpy(params[:3]).float()
    dims = torch.from_numpy(params[3:6]).float()
    quat = torch.from_numpy(params[6:10]).float()
    
    if mode == 'quat':
        # Quat → corners → quat
        corners = corners_from_params_torch(center, dims, quat).numpy()
        new_params = params_from_corners_quat(corners)
        new_quat = torch.from_numpy(new_params[6:10]).float()
        
        dot = torch.dot(quat, torch.tensor(new_quat).float()).abs()
        angle = (2 * torch.acos(dot.clamp(-1,1))) * (180 / torch.pi)
        if angle <= 1.0:  # 1 degree tolerance
            print(f"Successful! Angle: {angle}")
            return True
        else:
            print(f"Failed! Angle: {angle}")
            return False
    
    # Full mode
    reconstructed = corners_from_params_torch(center, dims, quat).numpy()
    from scipy.spatial.distance import cdist
    dists = cdist(corners_np, reconstructed)
    min_dists = np.min(dists, axis=1)
    mean_dist = np.mean(min_dists)

    # print("Original corners: ", corners_np)
    # print("Reconstructed corners: ", reconstructed)
    
    if mean_dist <= tolerance:
        print(f"Torch round-trip successful! Mean dist: {mean_dist}")
        return True
    else:
        print(f"Torch round-trip failed! Mean dist: {mean_dist}")
        return False
    

if __name__ == '__main__':
    
    folder = 'data'
    subfolder = random.choice(os.listdir(folder))

    test_paths = [
        {'image': f'{folder}/{subfolder}/rgb.jpg',
        'mask': f'{folder}/{subfolder}/mask.npy',
        'pc': f'{folder}/{subfolder}/pc.npy',
        'bbox': f'{folder}/{subfolder}/bbox3d.npy'},
    ]

    # 1. BBOX CONVERSION

    # Round-trip test on the first object's original corners
    # Load original GT bboxes from file (since dataset converts them to params)
    gt_bboxes = np.load(test_paths[0]['bbox'])  # (N, 8, 3)

    # Numpy version
    for i in range(len(gt_bboxes)):
        print(f"Testing object {i} in full mode (numpy):")
        test_bbox_conversion_numpy(gt_bboxes[i], mode='full')
        print(f"Testing object {i} in quat mode (numpy):")
        test_bbox_conversion_numpy(gt_bboxes[i], mode='quat')

    # Torch version
    for i in range(len(gt_bboxes)):
        print(f"Testing object {i} in full mode (Torch):")
        test_bbox_conversion_torch(gt_bboxes[i], mode='full')
        print(f"Testing object {i} in quat mode (Torch):")
        test_bbox_conversion_torch(gt_bboxes[i], mode='quat')


    # 2. RAW DATA

    visualize_raw_data_path(f'{folder}/{subfolder}', save_dir='vis/raw', name='test_raw')


    # 3. DATASET CREATION

    dataset = BBoxDataset(test_paths, max_points=MAX_POINTS, crop_size=CROP_SIZE)

    # Test single item
    item = dataset[0]  # First scene
    print("RGB Crops Shape:", item['rgb_crops'].shape)  # (N, 3, 224, 224)
    print("Points Shape:", item['points_list'].shape)  # (N, 1024, 3)
    print("Targets Shape:", item['targets'].shape)  # (N, 10)

    # Check for one object
    print("Sample Target Params:", item['targets'][0])  # Verify center, dims, quat look reasonable


    # 4. PROCESSED DATA VISUALIZATION

    visualize_processed_data_batch(item['rgb_crops'], item['points_list'], item['targets'],
        0, save_dir='vis/processed', name='test_processed')


    # 5. DATALOADER

    loader = DataLoader(dataset, batch_size=2, collate_fn=flatten_collate)  # Small batch
    batch = next(iter(loader))
    print("Batched RGB Shape:", batch['rgb_crops'].shape)  # e.g., (total_objects, 3, 224, 224)