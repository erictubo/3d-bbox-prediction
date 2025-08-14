import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from src.utils.data_utils import corners_from_params_batch

class HybridLoss(nn.Module):
    """
    Weighted hybrid loss combining params, corner distance, projected IoU, and angular quat.
    Adjustable weights; sums to 1.0.
    """
    def __init__(self, params_weight=0.0, corner_weight=1.0, iou_weight=0.0):
        super().__init__()
        assert corner_weight + iou_weight + params_weight == 1.0, "Weights must sum to 1.0"
        self.params_weight = params_weight
        self.corner_weight = corner_weight
        self.iou_weight = iou_weight

    def forward(self, pred_params: torch.Tensor, gt_params: torch.Tensor) -> torch.Tensor:

        total_loss = 0.0
        
        # Calculate gt_diagonals once for reuse
        gt_diagonals = get_diagonals(gt_params[:, 3:6])
        
        if self.params_weight > 0:
            center_loss_val = center_loss(pred_params[:, :3], gt_params[:, :3], gt_diagonals)
            dims_loss_val = dims_loss(pred_params[:, 3:6], gt_params[:, 3:6])
            angular_loss_val = angular_quat_loss(pred_params[:, 6:], gt_params[:, 6:])

            params_loss = center_loss_val + dims_loss_val + angular_loss_val
            total_loss += self.params_weight * params_loss

        if self.params_weight < 1.0:
            pred_corners = corners_from_params_batch(pred_params)
            gt_corners = corners_from_params_batch(gt_params)

            if self.corner_weight > 0:
                corner_loss = corner_distance_loss(pred_corners, gt_corners, gt_diagonals).mean()
                total_loss += self.corner_weight * corner_loss
            
            if self.iou_weight > 0:
                iou_loss = 1 - projected_iou_params(pred_params, gt_params).mean()
                total_loss += self.iou_weight * iou_loss
        
        return total_loss

    
    
def get_diagonals(dims: torch.Tensor) -> torch.Tensor:
    """
    Get the diagonal of a bounding box from its dimensions.
    """
    return torch.norm(dims, dim=-1, keepdim=True)
    
def center_loss(pred_center: torch.Tensor, gt_center: torch.Tensor, gt_diagonals: torch.Tensor) -> torch.Tensor:
    """
    Relative L1 loss on centers, normalized by gt diagonal.
    """
    rel_diff = torch.abs(pred_center - gt_center) / gt_diagonals
    return rel_diff.mean()

def dims_loss(pred_dims: torch.Tensor, gt_dims: torch.Tensor) -> torch.Tensor:
    """
    Relative L1 loss on dimensions, normalized by ground truth dims to keep it relative.
    Returns scalar mean loss.
    """
    rel_diff = torch.abs(pred_dims - gt_dims) / (gt_dims + 1e-6)
    return rel_diff.mean()

def quat_rotation_loss(pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
    """
    Rotation loss between predicted and ground truth quaternions (xyzw order).
    Handles sign ambiguity by considering minimal angle (abs dot product).
    Returns mean angular distance in radians.
    """
    pred_q = pred_q / (torch.norm(pred_q, dim=-1, keepdim=True) + 1e-6)
    gt_q = gt_q / (torch.norm(gt_q, dim=-1, keepdim=True) + 1e-6)
    
    # Abs to handle sign ambiguity (quats q and -q represent same rotation)
    dot = torch.abs(torch.einsum('bi,bi->b', pred_q, gt_q)).clamp(1e-6, 1.0)
    angle = 2 * torch.acos(dot)  # Angle in [0, pi]
    return angle.mean()











def angular_quat_loss_min8(pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
    pred_q = pred_q / (torch.norm(pred_q, dim=-1, keepdim=True) + 1e-6)
    gt_q = gt_q / (torch.norm(gt_q, dim=-1, keepdim=True) + 1e-6)

    # Generate 8 symmetries (sign combinations)
    signs = torch.tensor([[sx, sy, sz, sw] for sx in [1, -1] for sy in [1, -1] for sz in [1, -1] for sw in [1, -1]], device=pred_q.device).unsqueeze(0)  # (1,8,4)

    min_angles = torch.zeros(pred_q.size(0), device=pred_q.device)
    for b in range(pred_q.size(0)):
        gt_variants = gt_q[b:b+1] * signs  # (1,8,4) * (1,8,4) -> (1,8,4)
        gt_variants /= torch.norm(gt_variants, dim=-1, keepdim=True)
        dots = torch.abs(torch.einsum('bd,ikd->ik', pred_q[b:b+1], gt_variants)).clamp(1e-6, 1)
        angles = torch.acos(2 * dots**2 - 1)
        min_angles[b] = angles.min()

    return min_angles.mean()

def angular_quat_loss(pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
    """
    Angular loss between predicted and ground truth quaternions.
    Returns mean angular distance in radians.
    """
    pred_q = pred_q / (torch.norm(pred_q, dim=-1, keepdim=True) + 1e-6)
    gt_q = gt_q / (torch.norm(gt_q, dim=-1, keepdim=True) + 1e-6)
    
    # Abs to handle sign ambiguity (quats q and -q represent same rotation)
    dot = torch.abs(torch.einsum('bi,bi->b', pred_q, gt_q)).clamp(1e-6, 1.0)
    angle = 2 * torch.acos(dot)  # Angle in [0, pi]
    return angle.mean()






# def corner_distance_loss(pred_corners: torch.Tensor, gt_corners: torch.Tensor, gt_diagonals: torch.Tensor) -> torch.Tensor:
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

def corner_distance_loss(pred_corners: torch.Tensor, gt_corners: torch.Tensor, gt_diagonals: torch.Tensor) -> torch.Tensor:
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


# ----------------------------- TESTING -----------------------------

def test_losses():
    print("Benchmarking all losses...")
    
    pred_params = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Identical
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Disjoint
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0],  # Partial
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Rotated
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0],  # Sizes
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]   # Edge
    ], dtype=torch.float32)
    
    gt_params = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.7071, 0.0, 0.7071],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    expected_iou = [1.0, 0.0, 0.125, 0.707, 0.125, 0.0]  # For reference
    
    proj_iou = projected_iou_params(pred_params, gt_params, mode='mean')  # Use mean for better partials
    
    pred_corners = corners_from_params_batch(pred_params)
    gt_corners = corners_from_params_batch(gt_params)
    gt_diagonals = get_diagonals(gt_params[:, 3:6])
    cd_loss = corner_distance_loss(pred_corners, gt_corners, gt_diagonals)
        
    angular_loss = angular_quat_loss_min8(pred_params[:, 6:], gt_params[:, 6:])
    
    all_passed = True
    for i in range(pred_params.size(0)):
        case_name = ['Identical', 'Disjoint', 'Partial', 'Rotated', 'Sizes', 'Edge'][i]
        print(f"Case {i} ({case_name}):")
        print(f"  Expected IoU: {expected_iou[i]:.4f}")
        print(f"  Projected IoU: {proj_iou[i].item():.4f} (Dev: {abs(proj_iou[i] - expected_iou[i]):.4f})")
        print(f"  Corner Distance Loss: {cd_loss[i].item():.4f}")
        print(f"  Angular Quat Loss: {angular_loss:.4f}")
        if abs(proj_iou[i] - expected_iou[i]) > 0.1 or cd_loss[i] > 1.0:
            print("  ⚠ High deviation")
            all_passed = False
    
    print("✓ All accurate." if all_passed else "⚠ Some deviations—tune weights.")
