"""
3D Bounding Box Conversion Utilities

This module provides functions for converting between different 3D bounding box representations:
- Corner-based: 8 corners in 3D space (8, 3)
- Parameter-based: center (3), dimensions (3), quaternion (4) = 10 parameters total

The module supports both NumPy and PyTorch tensor operations, with Open3D integration
for robust oriented bounding box computation from corner points.

Key Functions:
- params_from_corners_quat: Convert 8 corners to center, dims, quaternion
- corners_from_params_numpy: Convert parameters to corners (NumPy)
- corners_from_params_torch: Convert parameters to corners (PyTorch)
- corners_from_params_batch: Batch version for PyTorch tensors

Testing in test_augmentations.py
"""

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d


def params_from_corners_quat(corners):
    """
    Convert 8 corners (8, 3) to center (3), dims (3), quat (4) using Open3D.
    Assumes corners are numpy array.
    """
    # Create point cloud from corners
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(corners)
    
    # Get oriented bounding box
    bbox = pcd.get_oriented_bounding_box()
    
    # Extract params
    center = np.asarray(bbox.center)
    dims = np.asarray(bbox.extent)
    rot_mat = np.asarray(bbox.R)
    
    # Convert rotation matrix to quaternion (x, y, z, w)
    # Make a copy to ensure it's writable
    rot_mat_copy = np.array(rot_mat, copy=True)
    rotation = R.from_matrix(rot_mat_copy)
    quat = rotation.as_quat()
    quat /= np.linalg.norm(quat)  # Ensure normalization
    
    return np.concatenate([center, dims, quat])

# def corners_from_params_quat(center, dims, quat):
#     """
#     Convert center (3), dims (3), quat (4) to 8 corners (8, 3) using Open3D.
#     All inputs as torch tensors.
#     """
#     # Convert to numpy for Open3D
#     center_np = center.numpy()
#     dims_np = dims.numpy()  # Extent in Open3D is full dimensions
#     rot_mat = R.from_quat(quat.numpy()).as_matrix()  # Quat to matrix
    
#     # Create oriented bbox
#     bbox = o3d.geometry.OrientedBoundingBox(center_np, rot_mat, dims_np)
    
#     # Get corners
#     corners_np = np.asarray(bbox.get_box_points())
    
#     return torch.from_numpy(corners_np).float()  # (8, 3)

def corners_from_params_numpy(center: np.ndarray, dims: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """
    Convert center (3), dims (3), quat (4) to 8 corners (8, 3) using NumPy.
    All inputs as NumPy arrays.
    """
    # Normalize quaternion
    quat = quat / np.linalg.norm(quat)
    
    # Quat to matrix (scalar last [x,y,z,w])
    x, y, z, w = quat
    rot_mat = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y + 2*z*w, 2*x*z - 2*y*w],
        [2*x*y - 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z + 2*x*w],
        [2*x*z + 2*y*w, 2*y*z - 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    
    # Create local corners (matching Open3D's box points order)
    half_dims = dims / 2
    local_corners = np.array([
        [-half_dims[0], -half_dims[1], -half_dims[2]],
        [-half_dims[0], -half_dims[1],  half_dims[2]],
        [-half_dims[0],  half_dims[1], -half_dims[2]],
        [-half_dims[0],  half_dims[1],  half_dims[2]],
        [ half_dims[0], -half_dims[1], -half_dims[2]],
        [ half_dims[0], -half_dims[1],  half_dims[2]],
        [ half_dims[0],  half_dims[1], -half_dims[2]],
        [ half_dims[0],  half_dims[1],  half_dims[2]]
    ])  # (8, 3)
    
    # Rotate and translate
    rotated = local_corners @ rot_mat
    corners = rotated + np.tile(center, (8, 1))
    return corners

def corners_from_params_torch(center: torch.Tensor, dims: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """
    Convert center (3), dims (3), quat (4) to 8 corners (8, 3) using PyTorch.
    All inputs as torch tensors.
    """
    
    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    
    # Quat to matrix (scalar last [x,y,z,w])
    x, y, z, w = quat.unbind(-1)
    rot_mat = torch.tensor([
        [1 - 2*y**2 - 2*z**2, 2*x*y + 2*z*w, 2*x*z - 2*y*w],
        [2*x*y - 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z + 2*x*w],
        [2*x*z + 2*y*w, 2*y*z - 2*x*w, 1 - 2*x**2 - 2*y**2]
    ], device=quat.device)
    
    # Create local corners (matching Open3D's box points order)
    half_dims = dims / 2
    local_corners = torch.tensor([
        [-half_dims[0], -half_dims[1], -half_dims[2]],
        [-half_dims[0], -half_dims[1],  half_dims[2]],
        [-half_dims[0],  half_dims[1], -half_dims[2]],
        [-half_dims[0],  half_dims[1],  half_dims[2]],
        [ half_dims[0], -half_dims[1], -half_dims[2]],
        [ half_dims[0], -half_dims[1],  half_dims[2]],
        [ half_dims[0],  half_dims[1], -half_dims[2]],
        [ half_dims[0],  half_dims[1],  half_dims[2]]
    ], device=center.device)  # (8, 3)
    
    # Rotate and translate
    rotated = torch.matmul(local_corners, rot_mat)
    corners = rotated + center.unsqueeze(0)
    return corners

def corners_from_params_batch(params: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of params (B, 10) to batch of corners (B, 8, 3).
    """
    assert params.shape[1] == 10, f"params shape: {params.shape}"
    corners = torch.stack([corners_from_params_torch(params[b, :3], params[b, 3:6], params[b, 6:]) for b in range(params.size(0))])
    assert corners.shape == (params.size(0), 8, 3), f"corners shape: {corners.shape}"
    return corners