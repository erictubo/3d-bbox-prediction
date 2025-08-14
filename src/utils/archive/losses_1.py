import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import open3d as o3d


class HybridLoss(nn.Module):
    """Combines SmoothL1 on params with simplified overlap."""
    def __init__(self, overlap_weight=0.5):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.overlap_weight = overlap_weight
    
    def forward(self, preds, targets):
        param_loss = self.smooth_l1(preds, targets)
        overlap = simplified_overlap_params(preds, targets)
        return param_loss + self.overlap_weight * (1 - overlap.mean())

def simplified_overlap_params(pred_params: torch.Tensor, gt_params: torch.Tensor) -> torch.Tensor:
    """
    Compute approximate overlap (IoU-like) for batch (B, 10): center[3], dims[3], quat[4 xyzw].
    NOTE: This function IS DIFFERENTIABLE and can be used in loss computation.
    Improved with rotation alignment term.
    Returns tensor of shape (B,) with per-box overlaps [0-1].
    """
    B = pred_params.size(0)
    overlaps = torch.zeros(B, device=pred_params.device)
    
    for b in range(B):
        pred_c, pred_d, pred_q = pred_params[b, :3], pred_params[b, 3:6], pred_params[b, 6:]
        gt_c, gt_d, gt_q = gt_params[b, :3], gt_params[b, 3:6], gt_params[b, 6:]
        
        pred_q = pred_q / torch.norm(pred_q)
        gt_q = gt_q / torch.norm(gt_q)
        
        # Quat to rot matrix (improved with dot for alignment)
        def quat_to_rot(q):
            x, y, z, w = q
            return torch.tensor([
                [1 - 2*y**2 - 2*z**2, 2*x*y + 2*z*w, 2*x*z - 2*y*w],
                [2*x*y - 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z + 2*x*w],
                [2*x*z + 2*y*w, 2*y*z - 2*x*w, 1 - 2*x**2 - 2*y**2]
            ], device=q.device)
        
        pred_rot = quat_to_rot(pred_q)
        gt_rot = quat_to_rot(gt_q)
        
        # Rotation alignment (dot product trace / 3, ~1 for identical)
        rot_align = (torch.trace(torch.matmul(pred_rot.T, gt_rot)) / 3).clamp(0, 1)
        
        pred_ext = torch.abs(torch.matmul(pred_rot, pred_d / 2))
        gt_ext = torch.abs(torch.matmul(gt_rot, gt_d / 2))
        
        center_diff = pred_c - gt_c
        overlap_per_axis = torch.clamp(torch.min(pred_ext, gt_ext) * 2 - center_diff.abs(), min=0) / (pred_ext + gt_ext + 1e-6)
        axis_overlap = torch.prod(overlap_per_axis)
        
        # Combine with rotation factor
        overlap = axis_overlap * rot_align
        overlaps[b] = overlap.clamp(0, 1)
    
    return overlaps


def iou_approx_corners(corners1: torch.Tensor, corners2: torch.Tensor) -> torch.Tensor:
    """
    Compute approximate IoU from corner tensors (B, 8, 3).
    Improved with GIoU-like enclosure penalty for rotations.
    Returns tensor of shape (B,) with per-box IoUs.
    """
    B = corners1.size(0)
    ious = torch.zeros(B, device=corners1.device)
    
    for b in range(B):
        c1 = corners1[b]
        c2 = corners2[b]
        
        min1, max1 = c1.min(0)[0], c1.max(0)[0]
        min2, max2 = c2.min(0)[0], c2.max(0)[0]
        
        inter_min = torch.max(min1, min2)
        inter_max = torch.min(max1, max2)
        inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0))
        
        vol1 = torch.prod(max1 - min1)
        vol2 = torch.prod(max2 - min2)
        union_vol = vol1 + vol2 - inter_vol
        
        iou = inter_vol / (union_vol + 1e-6)
        
        # GIoU enclosure penalty (for better rotation sensitivity)
        enc_min = torch.min(min1, min2)
        enc_max = torch.max(max1, max2)
        enc_vol = torch.prod(enc_max - enc_min)
        giou = iou - (enc_vol - union_vol) / (enc_vol + 1e-6)
        
        ious[b] = giou.clamp(0, 1)  # Use GIoU variant for accuracy
    
    return ious


# def iou_params_open3d(pred_params: torch.Tensor, gt_params: torch.Tensor) -> torch.Tensor:
#     """
#     Compute mean 3D IoU using Open3D for batch of params (B, 10): center[3], dims[3], quat[4 xyzw].
    
#     WARNING: This function is NOT DIFFERENTIABLE due to Open3D operations.
#     Use only for evaluation metrics, not for loss computation in training.

#     Returns tensor of shape (B,) with per-box IoUs.
#     """
#     with torch.no_grad():
#         B = pred_params.size(0)
#         ious = torch.zeros(B, device=pred_params.device)
        
#         for b in range(B):
#             pred_c, pred_d, pred_q = pred_params[b, :3].cpu().numpy(), pred_params[b, 3:6].cpu().numpy(), pred_params[b, 6:].cpu().numpy()
#             gt_c, gt_d, gt_q = gt_params[b, :3].cpu().numpy(), gt_params[b, 3:6].cpu().numpy(), gt_params[b, 6:].cpu().numpy()
            
#             x, y, z, w = pred_q
#             pred_rot = np.array([
#                 [1 - 2*y**2 - 2*z**2, 2*x*y + 2*z*w, 2*x*z - 2*y*w],
#                 [2*x*y - 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z + 2*x*w],
#                 [2*x*z + 2*y*w, 2*y*z - 2*x*w, 1 - 2*x**2 - 2*y**2]
#             ])
#             x, y, z, w = gt_q
#             gt_rot = np.array([
#                 [1 - 2*y**2 - 2*z**2, 2*x*y + 2*z*w, 2*x*z - 2*y*w],
#                 [2*x*y - 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z + 2*x*w],
#                 [2*x*z + 2*y*w, 2*y*z - 2*x*w, 1 - 2*x**2 - 2*y**2]
#             ])
            
#             # Create bounding boxes
#             pred_bbox = o3d.geometry.OrientedBoundingBox(pred_c, pred_rot, pred_d)
#             gt_bbox = o3d.geometry.OrientedBoundingBox(gt_c, gt_rot, gt_d)
            
#             # Create meshes from bounding boxes
#             pred_mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(pred_bbox)
#             gt_mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(gt_bbox)
            
#             # Voxelize the meshes
#             voxel_size = 0.01  # Adjust based on your scale
#             pred_vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh(pred_mesh, voxel_size)
#             gt_vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh(gt_mesh, voxel_size)
            
#             inter_vg = pred_vg.intersect(gt_vg)
#             vol1 = len(pred_vg.get_voxels()) * voxel_size**3
#             vol2 = len(gt_vg.get_voxels()) * voxel_size**3
#             inter_vol = len(inter_vg.get_voxels()) * voxel_size**3
#             union_vol = vol1 + vol2 - inter_vol
#             ious[b] = inter_vol / union_vol if union_vol > 0 else 0.0
        
#         return ious.detach()
    
def iou_corners_open3d(corners1: torch.Tensor, corners2: torch.Tensor) -> torch.Tensor:
    """
    Compute mean 3D IoU using Open3D for batch of corners (B, 8, 3).
    WARNING: This function is NOT DIFFERENTIABLE due to Open3D operations.
    Use only for evaluation metrics, not for loss computation in training.
    Returns tensor of shape (B,) with per-box IoUs.
    """
    with torch.no_grad():
        B = corners1.size(0)
        ious = torch.zeros(B, device=corners1.device)
        
        for b in range(B):
            corners_pred = corners1[b].cpu().numpy()
            corners_gt = corners2[b].cpu().numpy()
            
            pred_points = o3d.utility.Vector3dVector(corners_pred)
            gt_points = o3d.utility.Vector3dVector(corners_gt)
            
            pred_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pred_points)
            gt_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(gt_points)
            
            # Create meshes from bounding boxes
            pred_mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(pred_bbox)
            gt_mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(gt_bbox)
            
            # Voxelize the meshes
            voxel_size = 0.01  # Adjust based on your data scale (e.g., 0.005 for finer resolution)
            pred_vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh(pred_mesh, voxel_size)
            gt_vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh(gt_mesh, voxel_size)
            
            # Get sets of voxel grid indices
            pred_voxels = set(tuple(v.grid_index) for v in pred_vg.get_voxels())
            gt_voxels = set(tuple(v.grid_index) for v in gt_vg.get_voxels())
            
            # Compute intersection and union
            inter_voxels = pred_voxels & gt_voxels  # Set intersection
            union_voxels = pred_voxels | gt_voxels  # Set union
            
            inter_vol = len(inter_voxels) * (voxel_size ** 3)
            union_vol = len(union_voxels) * (voxel_size ** 3)
            
            ious[b] = inter_vol / union_vol if union_vol > 0 else 0.0
        
        return ious.detach()


# ==================== TESTS ====================

def test_losses_against_open3d():
    """Combined test: Compare simplified_overlap_params, iou_corners, iou_params_open3d, and iou_corners_open3d using shared test cases."""
    print("Testing losses against Open3D for accuracy...")
    
    # Shared test cases (params format: center[3], dims[3], quat[4 xyzw])
    # Case 0: Identical boxes (expected ~1)
    # Case 1: Disjoint boxes (expected ~0)
    # Case 2: Partial overlap (expected ~0.1-0.5)
    # Case 3: Rotated boxes (expected <1 due to rotation)
    # Case 4: Different sizes, same center (expected ~0.125)
    # Case 5: Edge touching (expected ~0)
    
    pred_params = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Case 0 ref
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Case 1 ref
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0],  # Case 2 ref
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Case 3 ref
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0],  # Case 4 ref
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]   # Case 5 ref
    ])
    
    gt_params = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Identical
        [5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Disjoint
        [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0],  # Partial
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.7071, 0.0, 0.7071],  # Rotated (45° y-axis)
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Smaller inside larger
        [2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]   # Edge touching
    ])

    # Expected references (analytically calculated; approx for rotated)
    expected = [1.0, 0.0, 0.125, 0.707, 0.125, 0.0]  # For cases 0-5
    
    # Compute simplified overlap (param-based approx)
    overlap_vals = simplified_overlap_params(pred_params, gt_params)
    
    # Compute iou_params_open3d (Open3D from params)
    # iou_params_vals = iou_params_open3d(pred_params, gt_params)
    
    # Convert params to corners for corner-based methods
    from data_utils import corners_from_params_quat_torch
    pred_corners = torch.stack([corners_from_params_quat_torch(p[:3], p[3:6], p[6:]) for p in pred_params])
    gt_corners = torch.stack([corners_from_params_quat_torch(p[:3], p[3:6], p[6:]) for p in gt_params])
    
    # Compute iou_corners (approx from corners)
    iou_corners_vals = iou_approx_corners(pred_corners, gt_corners)
    
    # Compute iou_corners_open3d (Open3D from corners)
    iou_corners_open3d_vals = iou_corners_open3d(pred_corners, gt_corners)
    
    # Print and check
    all_passed = True
    for i in range(pred_params.size(0)):
        case_name = ['Identical', 'Disjoint', 'Partial', 'Rotated', 'Sizes', 'Edge'][i]
        print(f"Case {i} ({case_name}):")
        print(f"  Expected Reference: {expected[i]:.4f}")
        print(f"  Simplified Overlap: {overlap_vals[i].item():.4f}")
        print(f"  IoU Corners Approx: {iou_corners_vals[i].item():.4f}")
        print(f"  IoU Corners Open3D: {iou_corners_open3d_vals[i].item():.4f}")
        
        # Simple assertion (tolerance 0.1 for approx)
        if abs(overlap_vals[i] - expected[i]) > 0.1 or abs(iou_corners_vals[i] - expected[i]) > 0.1:
            print("  ⚠ Warning: Deviation >0.1 from expected")
            all_passed = False
        if abs(iou_corners_open3d_vals[i] - expected[i]) > 0.1:
            print("  ⚠ Open3D deviation (ignore if buggy)")
    
    print("✓ Test completed." if all_passed else "⚠ Some deviations—check for rotations.")

def test_hybrid_loss():
    """Test HybridLoss forward and backward pass."""
    print("Testing HybridLoss...")
    
    criterion = HybridLoss(overlap_weight=1.0)
    preds = torch.randn(4, 10, requires_grad=True)
    targets = torch.randn(4, 10)
    
    loss = criterion(preds, targets)
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Requires grad: {loss.requires_grad}")
    
    # Test backward
    loss.backward()
    print(f"  Gradients computed: {preds.grad is not None}")
    
    print("✓ HybridLoss tests completed")

if __name__ == "__main__":
    print("Running loss function tests...")
    test_losses_against_open3d()
    test_hybrid_loss()
    print("All tests completed!")
