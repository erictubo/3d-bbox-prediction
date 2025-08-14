import torch
from src.dataset import BBoxDataset
from src.config import MAX_POINTS, CROP_SIZE
import os
import numpy as np
from src.utils.data_utils import test_round_trip_quat_numpy, test_round_trip_quat_torch

subfolder = '8b061a8e-9915-11ee-9103-bbb8eae05561'

# Small test paths (replace with 1-5 actual dicts from your data)
test_paths = [
    {'image': f'data_subset/{subfolder}/rgb.jpg',
     'mask': f'data_subset/{subfolder}/mask.npy',
     'pc': f'data_subset/{subfolder}/pc.npy',
     'bbox': f'data_subset/{subfolder}/bbox3d.npy'},
    # Add more if needed
]

dataset = BBoxDataset(test_paths, max_points=MAX_POINTS, crop_size=CROP_SIZE)

# Test single item
item = dataset[0]  # First scene
print("RGB Crops Shape:", item['rgb_crops'].shape)  # Should be (N, 3, 224, 224) where N=1-20
print("Points Shape:", item['points_list'].shape)  # (N, 1024, 3)
print("Targets Shape:", item['targets'].shape)  # (N, 10)

# Check for one object
print("Sample Target Params:", item['targets'][0])  # Verify center, dims, quat look reasonable



# Round-trip test on the first object's original corners
# Load original GT bboxes from file (since dataset converts them to params)
gt_bboxes = np.load(test_paths[0]['bbox'])  # (N, 8, 3)
sample_corners = gt_bboxes[0]  # First object (8, 3)




print("Testing first object in full mode:")
test_round_trip_quat_numpy(sample_corners, mode='full')
print("Testing first object in quat mode:")
test_round_trip_quat_numpy(sample_corners, mode='quat')

# Optionally test all objects in the scene with both modes
for i in range(len(gt_bboxes)):
    print(f"Testing object {i} in full mode:")
    test_round_trip_quat_numpy(gt_bboxes[i], mode='full')
    print(f"Testing object {i} in quat mode:")
    test_round_trip_quat_numpy(gt_bboxes[i], mode='quat')


# PyTorch-differentiable version
print("Testing first object in full mode (Torch):")
test_round_trip_quat_torch(sample_corners, mode='full')
print("Testing first object in quat mode (Torch):")
test_round_trip_quat_torch(sample_corners, mode='quat')

# For all objects:
for i in range(len(gt_bboxes)):
    print(f"Testing object {i} in full mode (Torch):")
    test_round_trip_quat_torch(gt_bboxes[i], mode='full')
    print(f"Testing object {i} in quat mode (Torch):")
    test_round_trip_quat_torch(gt_bboxes[i], mode='quat')





# If using DataLoader (with collate)
from torch.utils.data import DataLoader
from src.dataset import flatten_collate

loader = DataLoader(dataset, batch_size=2, collate_fn=flatten_collate)  # Small batch
batch = next(iter(loader))
print("Batched RGB Shape:", batch['rgb_crops'].shape)  # e.g., (total_objects, 3, 224, 224)
