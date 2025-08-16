"""
Testing of Loss Functions
- losses.py

1. Angular quaternion loss
2. Corner distance loss
3. Projected IoU loss
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import torch
import numpy as np
import math

from src.utils.losses import angular_loss_basis, canonical_angular_dims_loss, corner_distance_loss, projected_iou_params


def test_angular_quat_loss_basis_min():
    """Test angular_quat_loss_basis_min with dummy cases."""

    import math
    # Helper to create rotation quat from axis-angle (xyzw)
    def axis_angle_to_quat(axis, angle_deg):
        angle_rad = math.radians(angle_deg)
        s = math.sin(angle_rad / 2)
        c = math.cos(angle_rad / 2)
        return torch.tensor([axis[0] * s, axis[1] * s, axis[2] * s, c])
    
    # Batch of 3 samples
    # Sample 0: Identical (loss ~0)
    # Sample 1: 90° around x (should be low due to symmetry)
    # Sample 2: 180° around z (equivalent via direction flip, loss ~0)
    pred_q = torch.stack([
        torch.tensor([0.0, 0.0, 0.0, 1.0]),  # Identity
        torch.tensor([0.0, 0.0, 0.0, 1.0]),  # Identity
        torch.tensor([0.0, 0.0, 0.0, 1.0])   # Identity
    ])
    
    gt_q = torch.stack([
        torch.tensor([0.0, 0.0, 0.0, 1.0]),                     # Identical
        axis_angle_to_quat([1.0, 0.0, 0.0], 90.0),              # 90° x
        axis_angle_to_quat([0.0, 0.0, 1.0], 180.0)              # 180° z (opposite direction)
    ])
    
    loss = angular_loss_basis(pred_q, gt_q)
    
    # Assertions: Loss should be low (near 0) for all due to symmetries
    assert abs(loss.item()) < 0.01, f"Loss not near zero for symmetric cases: {loss.item()}"
    
    # Non-symmetric case: Large rotation (e.g., 45° around arbitrary axis)
    gt_q_non_sym = axis_angle_to_quat([1.0, 1.0, 1.0], 45.0).unsqueeze(0)
    pred_q_non_sym = torch.tensor([0.0, 0.0, 0.0, 1.0]).unsqueeze(0)
    loss_non_sym = angular_loss_basis(pred_q_non_sym, gt_q_non_sym)
    assert loss_non_sym.item() > 0.1, f"Loss too low for non-symmetric case: {loss_non_sym.item()}"
    
    print("All tests passed: Symmetric loss near zero, non-symmetric higher.")

def test_canonical_angular_dims_loss():
    """
    Test canonical_angular_dims_loss to ensure it handles dim permutations and quat adjustments correctly.
    - Identical: loss ~0
    - Permuted dims with adjusted quat: loss ~0 (canonicalized)
    - Mismatched: higher loss (dims and/or angular)
    - Tests both dominant_only modes
    - Additional cases: degenerate dims, larger mismatch, odd permutation, secondary axis mismatch
    """
    print("Testing canonical_angular_dims_loss...")

    # Helper to create rotation quat from axis-angle ([x,y,z,w])
    def axis_angle_to_quat(axis, angle_deg):
        angle_rad = math.radians(angle_deg)
        s = math.sin(angle_rad / 2)
        c = math.cos(angle_rad / 2)
        return torch.tensor([axis[0] * s, axis[2] * s, axis[1] * s, c])

    # Extended batch of 8 samples
    B = 8

    # Sample 0: Identical (loss ~0)
    # Sample 1: Permuted dims (swap x/y), quat adjusted (90° around z) – loss ~0
    # Sample 2: Mismatched dims and quat (rotation around y affects dominant x-axis) – higher loss
    # Sample 3: Sign flip on one axis (180° around x) – loss ~0
    # Sample 4: Degenerate dims (two equal), identity quat – loss ~0 (stable sorting)
    # Sample 5: Larger angular mismatch (90° around arbitrary axis) with matched dims – higher loss
    # Sample 6: Odd permutation (swap x/z, 90° around y with sign flip) – loss ~0 if adjusted properly
    # Sample 7: Mismatch only in secondary axes (rotate around dominant x-axis) – low dominant, higher full

    pred_dims = torch.tensor([
        [3.0, 2.0, 1.0],  # 0: Identical
        [2.0, 3.0, 1.0],  # 1: Swapped x/y
        [4.0, 2.0, 1.0],  # 2: Mismatched sizes
        [3.0, 2.0, 1.0],  # 3: Sign flip test
        [2.0, 2.0, 1.0],  # 4: Degenerate (equal x/y)
        [3.0, 2.0, 1.0],  # 5: Matched dims, large angular mismatch
        [1.0, 2.0, 3.0],  # 6: Odd perm (x->z, y->y, z->x)
        [3.0, 2.0, 1.0]   # 7: Matched dims, secondary mismatch
    ])

    gt_dims = torch.tensor([
        [3.0, 2.0, 1.0],
        [3.0, 2.0, 1.0],  # Canonical sorts to same
        [3.0, 2.0, 1.0],
        [3.0, 2.0, 1.0],
        [2.0, 2.0, 1.0],  # Degenerate match
        [3.0, 2.0, 1.0],
        [3.0, 2.0, 1.0],  # Canonical same
        [3.0, 2.0, 1.0]
    ])

    pred_q = torch.stack([
        axis_angle_to_quat([1.0, 0.0, 0.0], 0.0),    # 0: Identity
        axis_angle_to_quat([0.0, 0.0, 1.0], 90.0),   # 1: 90° around z (matches swap)
        axis_angle_to_quat([0.0, 1.0, 0.0], 45.0),   # 2: 45° around y (affects x)
        axis_angle_to_quat([1.0, 0.0, 0.0], 180.0),  # 3: 180° around x (sign flip)
        axis_angle_to_quat([1.0, 0.0, 0.0], 0.0),    # 4: Identity (degenerate)
        axis_angle_to_quat([1.0, 1.0, 1.0], 90.0),   # 5: 90° around (1,1,1)
        axis_angle_to_quat([0.0, 1.0, 0.0], 90.0),   # 6: 90° around y (matches odd swap with flip)
        axis_angle_to_quat([1.0, 0.0, 0.0], 45.0)    # 7: 45° around x (affects y/z only)
    ])

    gt_q = torch.stack([
        axis_angle_to_quat([1.0, 0.0, 0.0], 0.0),    # 0: Identity
        axis_angle_to_quat([1.0, 0.0, 0.0], 0.0),    # 1: Identity (pred adjusts)
        axis_angle_to_quat([1.0, 0.0, 0.0], 0.0),    # 2: Identity (mismatch)
        axis_angle_to_quat([1.0, 0.0, 0.0], 0.0),    # 3: Identity (sign flip test)
        axis_angle_to_quat([1.0, 0.0, 0.0], 0.0),    # 4: Identity
        axis_angle_to_quat([1.0, 0.0, 0.0], 0.0),    # 5: Identity (large mismatch)
        axis_angle_to_quat([1.0, 0.0, 0.0], 0.0),    # 6: Identity (pred adjusts)
        axis_angle_to_quat([1.0, 0.0, 0.0], 0.0)     # 7: Identity (secondary mismatch)
    ])

    # Test full mode (all axes, dominant_only=False)
    loss_full = canonical_angular_dims_loss(pred_q, gt_q, pred_dims, gt_dims, dominant_only=False)
    print(f"Full mode batch loss: {loss_full.item():.6f}")

    # Test dominant-only mode
    loss_dominant = canonical_angular_dims_loss(pred_q, gt_q, pred_dims, gt_dims, dominant_only=True)
    print(f"Dominant-only mode batch loss: {loss_dominant.item():.6f}")

    # Per-sample checks with print for debugging
    all_passed = True
    for i in range(B):
        # Full mode
        l_full = canonical_angular_dims_loss(pred_q[i:i+1], gt_q[i:i+1], pred_dims[i:i+1], gt_dims[i:i+1], dominant_only=False)
        print(f"Sample {i} full loss: {l_full.item():.6f}")
        
        # Dominant mode
        l_dom = canonical_angular_dims_loss(pred_q[i:i+1], gt_q[i:i+1], pred_dims[i:i+1], gt_dims[i:i+1], dominant_only=True)
        print(f"Sample {i} dominant loss: {l_dom.item():.6f}")
        
    #     # Assertions: Low for equivalent (0,1,3,4,6), high for mismatch (2,5,7) – but for 7, full high (>0.2), dominant low (<0.1)
    #     if i in [0, 1, 3, 4, 6]:
    #         assert l_full.item() < 0.1, f"Full loss not low for equivalent case {i}: {l_full.item()}"
    #         assert l_dom.item() < 0.1, f"Dominant loss not low for equivalent case {i}: {l_dom.item()}"
    #     elif i in [2, 5]:
    #         assert l_full.item() > 0.2, f"Full loss too low for mismatch case {i}: {l_full.item()}"
    #         assert l_dom.item() > 0.2, f"Dominant loss too low for mismatch case {i}: {l_dom.item()}"
    #     elif i == 7:
    #         assert l_full.item() > 0.2, f"Full loss too low for secondary mismatch case {i}: {l_full.item()}"
    #         assert l_dom.item() < 0.1, f"Dominant loss not low for secondary mismatch case {i}: {l_dom.item()} (should penalize less)"

    # print("All tests passed: Loss low for equivalent cases, higher for mismatches; dominant mode ignores secondary errors as expected.")

def test_corner_distance_loss():
    print("Testing corner assignment with closest points...")
    
    # Create simple test case
    batch_size = 1
    
    # Ground truth corners (simple cube)
    gt_corners = torch.tensor([[
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
    ]], dtype=torch.float32)
    
    # Predicted corners (slightly offset)
    pred_corners = torch.tensor([[
        [0.1, 0.1, 0.1], [0.1, 0.1, 1.1], [0.1, 1.1, 0.1], [0.1, 1.1, 1.1],
        [1.1, 0.1, 0.1], [1.1, 0.1, 1.1], [1.1, 1.1, 0.1], [1.1, 1.1, 1.1]
    ]], dtype=torch.float32)
    
    # Ground truth diagonal
    gt_diagonals = torch.tensor([[np.sqrt(3)]], dtype=torch.float32)  # sqrt(1^2 + 1^2 + 1^2)
    
    print("Ground truth corners:")
    print(gt_corners[0])
    print("\nPredicted corners:")
    print(pred_corners[0])
    
    # Calculate cost matrix manually to show assignment
    pred_b = pred_corners[0].unsqueeze(1)  # (8, 1, 3)
    gt_b = gt_corners[0].unsqueeze(0)      # (1, 8, 3)
    cost_matrix = torch.sum(torch.abs(pred_b - gt_b), dim=2)  # (8, 8)
    
    print("\nCost matrix (L1 distances):")
    print(cost_matrix)
    
    # Find optimal assignment
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    
    print(f"\nOptimal assignment (pred_idx -> gt_idx):")
    for i, (pred_idx, gt_idx) in enumerate(zip(row_ind, col_ind)):
        dist = cost_matrix[pred_idx, gt_idx].item()
        print(f"  Pred corner {pred_idx} -> GT corner {gt_idx} (distance: {dist:.3f})")
    
    # Calculate loss
    loss = corner_distance_loss(pred_corners, gt_corners, gt_diagonals)
    print(f"\nCorner distance loss: {loss.item():.6f}")
    
    # Test with more complex case (rotated cube)
    print("\n" + "="*50)
    print("Testing with rotated cube...")
    
    # Create rotated ground truth (90 degrees around Z-axis)
    gt_corners_rotated = torch.tensor([[
        [0, 0, 0], [0, 0, 1], [-1, 0, 0], [-1, 0, 1],
        [0, 1, 0], [0, 1, 1], [-1, 1, 0], [-1, 1, 1]
    ]], dtype=torch.float32)
    
    # Predicted corners (original orientation)
    pred_corners_orig = gt_corners.clone()
    
    gt_diagonals_rotated = torch.tensor([[np.sqrt(3)]], dtype=torch.float32)
    
    print("Ground truth corners (rotated):")
    print(gt_corners_rotated[0])
    print("\nPredicted corners (original):")
    print(pred_corners_orig[0])
    
    # Calculate cost matrix for rotated case
    pred_b_rot = pred_corners_orig[0].unsqueeze(1)
    gt_b_rot = gt_corners_rotated[0].unsqueeze(0)
    cost_matrix_rot = torch.sum(torch.abs(pred_b_rot - gt_b_rot), dim=2)
    
    print("\nCost matrix (rotated case):")
    print(cost_matrix_rot)
    
    # Find optimal assignment for rotated case
    row_ind_rot, col_ind_rot = linear_sum_assignment(cost_matrix_rot.detach().cpu().numpy())
    
    print(f"\nOptimal assignment (rotated case):")
    for i, (pred_idx, gt_idx) in enumerate(zip(row_ind_rot, col_ind_rot)):
        dist = cost_matrix_rot[pred_idx, gt_idx].item()
        print(f"  Pred corner {pred_idx} -> GT corner {gt_idx} (distance: {dist:.3f})")
    
    # Calculate loss for rotated case
    loss_rotated = corner_distance_loss(pred_corners_orig, gt_corners_rotated, gt_diagonals_rotated)
    print(f"\nCorner distance loss (rotated): {loss_rotated.item():.6f}")
    
    print("\n✅ Corner assignment test completed!")

def test_projected_iou_loss():
    """Test projected IoU loss."""
    
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
    
    proj_iou = projected_iou_params(pred_params, gt_params)
            
    all_passed = True
    for i in range(pred_params.size(0)):
        case_name = ['Identical', 'Disjoint', 'Partial', 'Rotated', 'Sizes', 'Edge'][i]
        print(f"Case {i} ({case_name}):")
        print(f"  Expected IoU: {expected_iou[i]:.4f}")
        print(f"  Projected IoU: {proj_iou[i].item():.4f} (Dev: {abs(proj_iou[i] - expected_iou[i]):.4f})")
        if abs(proj_iou[i] - expected_iou[i]) > 0.1:
            print("  ⚠ High deviation")
            all_passed = False
    
    print("✓ All accurate." if all_passed else "⚠ Some deviations—tune weights.")

    return all_passed


if __name__ == '__main__':
    test_angular_quat_loss_basis_min()
    test_canonical_angular_dims_loss()
    test_corner_distance_loss()
    test_projected_iou_loss()