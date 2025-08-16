"""
Loss Functions

This module provides loss functions for training and evaluating 3D bounding box predictions.
It handles both losses based on parameters (center, dims, quat) and corners (8x 3D points).

Key Functions:
- HybridLoss: Weighted hybrid loss combining diagonal, center, quat, corner distance, and projected IoU.
- diagonal_loss: Relative L1 loss on dimensions, normalized by ground truth dims to keep it relative.
- center_loss: Relative L1 loss on centers, normalized by gt diagonal.
- angular_loss: Minimal angular loss using rotated basis vectors.
- canonical_angular_dims_loss: Canonicalized angular loss using rotated basis vectors in dims order.
- corner_distance_loss: Vectorized L1 corner distance loss with rotation penalty, normalized by gt diagonal.
- projected_iou_corners: Simple multi-plane projected IoU (xz, xy, yz; min average).
- projected_iou_params: Projected IoU loss for params-based predictions.

Testing in test_losses.py
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from src.utils.bbox_conversion import corners_from_params_batch


class HybridLoss(nn.Module):
    """
    Weighted hybrid loss combining diagonal, center, quat, corner distance, and projected IoU.
    Adjustable weights; sums to 1.0.
    """
    def __init__(self, loss_weights: dict[str, float]):
        super().__init__()
        assert sum(loss_weights.values()) == 1.0, "Weights must sum to 1.0"
        assert all(key in loss_weights for key in ['center', 'diag', 'quat', 'canonical', 'corner', 'iou'])
        self.center_weight = loss_weights['center']
        self.diag_weight = loss_weights['diag']
        self.quat_weight = loss_weights['quat']
        self.canonical_weight = loss_weights['canonical']
        self.corner_weight = loss_weights['corner']
        self.iou_weight = loss_weights['iou']

    def forward(self, pred_params: torch.Tensor, gt_params: torch.Tensor) -> torch.Tensor:

        loss_dict = {
            'center': None,
            'diag': None,
            'quat': None,
            'canonical': None,
            'corner': None,
            'iou': None
        }
        
        total_loss = 0.0
        
        # Calculate gt_diagonals once for reuse
        gt_diagonals = get_diagonals(gt_params[:, 3:6])

        if self.diag_weight > 0:
            diag_loss_val = diagonal_loss(pred_params[:, 3:6], gt_params[:, 3:6])
            loss_dict['diag'] = diag_loss_val.item()
            total_loss += self.diag_weight * diag_loss_val
        
        if self.center_weight > 0:
            center_loss_val = center_loss(pred_params[:, :3], gt_params[:, :3], gt_diagonals)
            loss_dict['center'] = center_loss_val.item()
            total_loss += self.center_weight * center_loss_val

        if self.quat_weight > 0:
            quat_loss_val = quat_loss(pred_params[:, 6:], gt_params[:, 6:])
            loss_dict['quat'] = quat_loss_val.item()
            total_loss += self.quat_weight * quat_loss_val

        # Canonical loss:
        if self.canonical_weight > 0:
            canonical_loss_val = canonical_dims_quat_loss(pred_params[:, 3:6], gt_params[:, 3:6], pred_params[:, 6:], gt_params[:, 6:])
            loss_dict['canonical'] = canonical_loss_val.item()
            total_loss += self.canonical_weight * canonical_loss_val

        # Convert to corners only if necessary for loss computation
        if self.corner_weight > 0 or self.iou_weight > 0:
            pred_corners = corners_from_params_batch(pred_params)
            gt_corners = corners_from_params_batch(gt_params)

            # Corner loss
            if self.corner_weight > 0:
                corner_loss_val = corner_distance_loss(pred_corners, gt_corners, gt_diagonals).mean()
                loss_dict['corner'] = corner_loss_val.item()
                total_loss += self.corner_weight * corner_loss_val
            
            # IoU loss
            if self.iou_weight > 0:
                iou_loss_val = 1 - projected_iou_params(pred_params, gt_params).mean()
                loss_dict['iou'] = iou_loss_val.item()
                total_loss += self.iou_weight * iou_loss_val
        
        return total_loss, loss_dict


# PARAMS-BASED LOSSES
    
def get_diagonals(dims: torch.Tensor) -> torch.Tensor:
    """
    Get the diagonal of a bounding box from its dimensions.
    """
    return torch.norm(dims, dim=-1, keepdim=True)

def diagonal_loss(pred_diag: torch.Tensor, gt_diag: torch.Tensor) -> torch.Tensor:
    """
    Relative L1 loss on dimensions, normalized by ground truth dims to keep it relative.
    """
    rel_diff = torch.abs(pred_diag - gt_diag) / (gt_diag + 1e-6)  # (B, 1)
    return rel_diff.mean()
    
def center_loss(pred_center: torch.Tensor, gt_center: torch.Tensor, gt_diagonals: torch.Tensor) -> torch.Tensor:
    """
    Relative L1 loss on centers, normalized by gt diagonal.
    """
    rel_diff = torch.abs(pred_center - gt_center) / gt_diagonals
    return rel_diff.mean()

def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """Convert quat (B,4) to rotation matrix (B,3,3)."""
    B = q.size(0)
    x, y, z, w = q[:,0], q[:,1], q[:,2], q[:,3]
    mat = torch.stack([
            1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w,
            2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
            2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2
        ], dim=1).view(B, 3, 3).to(q.device)
    return mat

def quat_loss(pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
    """
    Minimal angular loss using rotated basis vectors.
    Applies pred/gt quats to standard xyz basis, computes minimal angle over axis permutations
    (considering direction equivalence via abs dot).
    
    - pred_q: (B, 4) predicted quaternions (xyzw)
    - gt_q: (B, 4) ground truth quaternions (xyzw)
    
    Returns mean minimal angular distance in radians over batch.
    """
    B = pred_q.size(0)
    
    # Normalize quaternions
    pred_q = pred_q / (torch.norm(pred_q, dim=-1, keepdim=True) + 1e-6)
    gt_q = gt_q / (torch.norm(gt_q, dim=-1, keepdim=True) + 1e-6)
    
    # Standard basis vectors (x,y,z)
    basis = torch.eye(3, device=pred_q.device).unsqueeze(0)  # (1, 3, 3)    
    
    pred_mat = quat_to_mat(pred_q)  # (B,3,3)
    gt_mat = quat_to_mat(gt_q)      # (B,3,3)
    
    # Rotate basis: (B,3,3) @ (1,3,3) -> (B,3,3) [each col is rotated basis vector]
    pred_basis = torch.bmm(pred_mat, basis.expand(B, -1, -1))  # (B,3,3)
    gt_basis = torch.bmm(gt_mat, basis.expand(B, -1, -1))      # (B,3,3)
    
    min_angles = torch.zeros(B, device=pred_q.device)
    
    for b in range(B):
        # For each pred axis, find best match to gt axes (abs dot for direction equiv)
        dots = torch.abs(torch.matmul(pred_basis[b].T, gt_basis[b]))  # (3,3) abs dots between axes
        # Min angle per pred axis (max dot = min angle), then average over axes for sample loss
        max_dots = dots.max(dim=1)[0]  # Best match for each pred axis (3,)
        angles = torch.acos(max_dots.clamp(1e-6, 1.0))  # (3,)
        min_angles[b] = angles.mean()  # Mean angle over 3 axes (or min for stricter)
    
    return min_angles.mean()  # Batch mean

def canonical_dims_quat_loss(
    pred_dims: torch.Tensor, 
    gt_dims: torch.Tensor,
    pred_q: torch.Tensor, 
    gt_q: torch.Tensor, 
    dims_weight: float = 0.5,
    angular_weight: float = 0.5,
    dominant_only: bool = False
) -> torch.Tensor:
    """
    Combined loss for rearranged dimensions and canonicalized quaternions.
    
    - Sorts dims descending for both pred and gt to canonical order.
    - Computes relative L1 loss on sorted dims (normalized by gt sorted dims).
    - Reorders standard basis vectors according to the dim permutations 
      (associating basis with sorted dims: largest dim -> x, etc.).
    - Rotates the reordered bases using pred/gt quaternions.
    - Computes angular loss between corresponding rotated bases (one-to-one after canonicalization,
      using abs dot for direction equiv). If dominant_only=True, compares only the largest dim's axis.
    
    Inputs:
      pred_q, gt_q: (B, 4) quaternions [x,y,z,w]
      pred_dims, gt_dims: (B, 3) dimension vectors
      dims_weight, angular_weight: weights for combining losses (sum to 1.0)
      dominant_only: If True, only compare the axis for the largest dim; else, average over all three.
    
    Returns: weighted scalar loss (dims + angular in radians)
    """
    assert dims_weight + angular_weight == 1.0, "Weights must sum to 1.0"
    B = pred_q.size(0)
    
    # Normalize quats
    pred_q = pred_q / (torch.norm(pred_q, dim=-1, keepdim=True) + 1e-8)
    gt_q = gt_q / (torch.norm(gt_q, dim=-1, keepdim=True) + 1e-8)
    
    # Sort dims descending and get permutations
    pred_sorted_dims, pred_perm_idx = pred_dims.sort(dim=1, descending=True)
    gt_sorted_dims, gt_perm_idx = gt_dims.sort(dim=1, descending=True)
    
    # Dims loss on sorted dims (relative L1, normalized by gt)
    rel_diff = torch.abs(pred_sorted_dims - gt_sorted_dims) / (gt_sorted_dims + 1e-8)
    dims_loss = rel_diff.mean()
    
    # Standard basis vectors (xyz)
    basis = torch.eye(3, device=pred_q.device).unsqueeze(0).repeat(B, 1, 1)  # (B, 3, 3)
    
    # Rearrange the basis in pred and gt according to the dim order permutation
    pred_basis_reordered = torch.stack([basis[b, pred_perm_idx[b]] for b in range(B)])  # (B,3,3)
    gt_basis_reordered = torch.stack([basis[b, gt_perm_idx[b]] for b in range(B)])      # (B,3,3)
    
    # Convert quats to rotation matrices
    pred_rot = quat_to_mat(pred_q)  # (B,3,3)
    gt_rot = quat_to_mat(gt_q)      # (B,3,3)
    
    # Rotate the reordered basis vectors by their quats
    pred_rotated_basis = torch.bmm(pred_rot, pred_basis_reordered)  # (B,3,3)
    gt_rotated_basis = torch.bmm(gt_rot, gt_basis_reordered)        # (B,3,3)
    
    angular_losses = []
    for b in range(B):
        pred_vecs = pred_rotated_basis[b]  # (3,3) each row a vector
        gt_vecs = gt_rotated_basis[b]      # (3,3)
        
        # Compute absolute dot products between corresponding axes (one-to-one)
        abs_dots = torch.abs(torch.sum(pred_vecs * gt_vecs, dim=1))  # (3,)
        
        # Calculate angles for corresponding pairs
        angles = torch.acos(torch.clamp(abs_dots, 0, 1))  # (3,)
        
        if dominant_only:
            # Only use the angle for the dominant (first/largest dim) axis
            angular_losses.append(angles[0])
        else:
            # Average over all three axes
            angular_losses.append(angles.mean())
    
    angular_loss = torch.stack(angular_losses).mean()
    
    # Weighted combination
    total_loss = dims_weight * dims_loss + angular_weight * angular_loss
    return total_loss



# CORNER-BASED LOSSES

def corner_distance_loss(pred_corners: torch.Tensor, gt_corners: torch.Tensor, gt_diagonals: torch.Tensor) -> torch.Tensor:
#     """
#     Vectorized L1 corner distance loss with rotation penalty, normalized by gt diagonal.
#     Uses Hungarian matching for robustness.
#     Returns (B,) tensor of losses.
#     """
#     B = pred_corners.size(0)
#     assert pred_corners.shape == (B, 8, 3), f"pred_corners shape: {pred_corners.shape}"
#     assert gt_corners.shape == (B, 8, 3), f"gt_corners shape: {gt_corners.shape}"
#     assert gt_diagonals.shape == (B, 1), f"gt_diagonals shape: {gt_diagonals.shape}"

#     batch_losses = torch.zeros(B, device=pred_corners.device)

#     for b in range(B):
#         cost = torch.cdist(pred_corners[b], gt_corners[b], p=1)  # L1 matrix (8,8)
#         row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
#         dists = cost[row_ind, col_ind]

#         norm_dists = dists / (gt_diagonals[b] + 1e-6)
#         batch_losses[b] = norm_dists.mean()

#     return batch_losses
    """
    Vectorized L1 corner distance loss with rotation penalty, normalized by gt diagonal.
    Uses custom broadcasting for pairwise L1 (MPS-compatible, avoids cdist).
    Returns (B,) tensor of losses.
    """
    B = pred_corners.size(0)
    assert pred_corners.shape == (B, 8, 3), f"pred_corners shape: {pred_corners.shape}"
    assert gt_corners.shape == (B, 8, 3), f"gt_corners shape: {gt_corners.shape}"
    assert gt_diagonals.shape == (B, 1) or gt_diagonals.shape == (B,), f"gt_diagonals shape: {gt_diagonals.shape}"

    # Handle gt_diagonals shape: squeeze if (B, 1) to (B,)
    if gt_diagonals.shape == (B, 1):
        gt_diagonals = gt_diagonals.squeeze(-1)
    
    batch_losses = torch.zeros(B, device=pred_corners.device)

    for b in range(B):
        # # Custom L1 distance matrix (MPS-friendly broadcasting): (8,1,3) - (1,8,3) -> abs -> sum
        # diffs = torch.abs(pred_corners[b].unsqueeze(1) - gt_corners[b].unsqueeze(0))  # (8,8,3)
        # cost = diffs.sum(dim=-1)  # (8,8) L1 distances

        # # Hungarian assignment
        # row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())  # SciPy is CPU-only, short op
        # dists = cost[row_ind, col_ind]  # Back to tensor

        # norm_dists = dists / (gt_diagonals[b] + 1e-6)
        # loss = norm_dists.mean()

        # batch_losses[b] = loss  # Removed rotation penalty as per your update

        # Move to CPU for computation (avoids MPS slice error)
        pred_b = pred_corners[b].cpu()
        gt_b = gt_corners[b].cpu()

        # Custom L1 distance matrix on CPU
        diffs = torch.abs(pred_b.unsqueeze(1) - gt_b.unsqueeze(0))  # (8,8,3)
        cost = diffs.sum(dim=-1)  # (8,8) L1 distances

        # Hungarian on CPU
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        dists = cost[row_ind, col_ind]

        # Normalize and back to device
        norm_dists = dists / (gt_diagonals[b].cpu() + 1e-6)
        loss = norm_dists.mean()

        batch_losses[b] = loss.to(pred_corners.device)  # Back to MPS/CPU

    return batch_losses


def projected_iou_corners(corners1: torch.Tensor, corners2: torch.Tensor, eps=1e-3, mode='mean') -> torch.Tensor:
    """
    Simple multi-plane projected IoU (xz, xy, yz; min average).
    """
    B = corners1.size(0)
    planes = torch.tensor([[0,2], [0,1], [1,2]], device=corners1.device)
    perp_axes = torch.tensor([1, 2, 0], device=corners1.device)
    
    proj1 = corners1[:, :, planes].permute(0, 2, 1, 3)  # (B, 3, 8, 2)
    proj2 = corners2[:, :, planes].permute(0, 2, 1, 3)
    
    min1, max1 = proj1.amin(dim=2), proj1.amax(dim=2)
    min2, max2 = proj2.amin(dim=2), proj2.amax(dim=2)
    
    inter_min = torch.maximum(min1, min2)
    inter_max = torch.minimum(max1, max2)
    inter_area = torch.prod(torch.clamp(inter_max - inter_min + eps, min=0), dim=-1)
    
    area1 = torch.prod(max1 - min1 + eps, dim=-1)
    area2 = torch.prod(max2 - min2 + eps, dim=-1)
    union_area = area1 + area2 - inter_area
    iou_2d = inter_area / (union_area + 1e-6)
    
    h1_min = corners1[:, :, perp_axes].amin(dim=1)
    h1_max = corners1[:, :, perp_axes].amax(dim=1)
    h2_min = corners2[:, :, perp_axes].amin(dim=1)
    h2_max = corners2[:, :, perp_axes].amax(dim=1)
    h_inter = torch.clamp(torch.minimum(h1_max, h2_max) - torch.maximum(h1_min, h2_min), min=0)
    h_union = (h1_max - h1_min) + (h2_max - h2_min) - h_inter
    iou_h = h_inter / (h_union + 1e-6)
    
    iou_per_plane = iou_2d * iou_h
    if mode == 'mean':
        return iou_per_plane.mean(dim=1).clamp(0, 1)
    elif mode == 'min':
        return iou_per_plane.min(dim=1)[0].clamp(0, 1)
    else:
        raise ValueError(f"Invalid mode: {mode}")

def projected_iou_params(pred_params: torch.Tensor, gt_params: torch.Tensor) -> torch.Tensor:
    pred_corners = corners_from_params_batch(pred_params)
    gt_corners = corners_from_params_batch(gt_params)
    return projected_iou_corners(pred_corners, gt_corners)