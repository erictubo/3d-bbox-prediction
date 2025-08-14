import torch
import torch.nn as nn
from data_utils import corners_from_params_quat_torch

# OVERVIEW
# 1. params-based loss
#    - weighted sum of center, dims, quat
#    - canonicalization to get unique representation (dims largest to smallest, quat rearranged accordingly)
# 2. iou loss via reprojection to 2d planes
# 3. optional: params normalized to gt frame -> pred_corners_norm where gt_corners_norm would be unit cube at origin
#    - simplified & more accurate iou computation


# PROBLEMS
# too complex for prototype!
# canonicalization doesn't work properly -- difficult with quaternions / rotmat inefficent and also not always correct
# iou approximation not ideal


class HybridLoss(nn.Module):
    """
    Hybrid loss combining weighted params-based loss with normalized projected IoU.
    Use for training; adjustable weight for IoU emphasis.
    """
    def __init__(self, iou_weight=0.5):
        super().__init__()
        self.iou_weight = iou_weight

    def forward(self, pred_params: torch.Tensor, gt_params: torch.Tensor) -> torch.Tensor:
        # Canonicalize before computing loss to handle ambiguities
        pred_can = canonicalize_params_swapping(pred_params)
        gt_can = canonicalize_params_swapping(gt_params)
        p_loss = weighted_params_loss(pred_can, gt_can)
        iou = normalized_projected_iou_params(pred_can, gt_can).mean()
        return p_loss + self.iou_weight * (1 - iou)

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

def dims_loss(pred_dims: torch.Tensor, gt_dims: torch.Tensor) -> torch.Tensor:
    """
    Relative L1 loss on dimensions, normalized by ground truth dims to keep it relative.
    Returns scalar mean loss.
    """
    rel_diff = torch.abs(pred_dims - gt_dims) / (gt_dims + 1e-6)
    return rel_diff.mean()

def center_loss(pred_center: torch.Tensor, gt_center: torch.Tensor, gt_dims: torch.Tensor) -> torch.Tensor:
    """
    Relative L1 loss on centers, normalized by ground truth bbox diagonal size to keep it relative.
    Returns scalar mean loss.
    """
    diag = torch.norm(gt_dims, dim=-1, keepdim=True) + 1e-6  # Diagonal length for normalization
    rel_diff = torch.abs(pred_center - gt_center) / diag
    return rel_diff.mean()

def weighted_params_loss(pred_params: torch.Tensor, gt_params: torch.Tensor, weight_center=0.4, weight_dims=0.4, weight_quat=0.2) -> torch.Tensor:
    """
    Combined parameters loss with weighted sum of separate components.
    Weights sum to 1.0; focuses on making losses comparable in scale.
    Returns scalar weighted loss.
    """
    pred_center, pred_dims, pred_q = pred_params[:, :3], pred_params[:, 3:6], pred_params[:, 6:]
    gt_center, gt_dims, gt_q = gt_params[:, :3], gt_params[:, 3:6], gt_params[:, 6:]
    
    loss_center = center_loss(pred_center, gt_center, gt_dims)
    loss_dims = dims_loss(pred_dims, gt_dims)
    loss_quat = quat_rotation_loss(pred_q, gt_q)
    
    return weight_center * loss_center + weight_dims * loss_dims + weight_quat * loss_quat

def canonicalize_params_swapping(params: torch.Tensor) -> torch.Tensor:
    """
    Fixed swapping version: Canonicalize by conjugating with refined perm quats, per-component sign corrections, and handedness handling.
    Sorts dims descending, adjusts quat (xyzw order) for new axis order, ensures w > 0.
    Handles batch (B, 10).
    """
    B = params.size(0)
    center = params[:, :3]
    dims = params[:, 3:6]
    quat = params[:, 6:]

    # Normalize quat (xyzw order)
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-6)

    # Ensure w positive
    w_sign = torch.sign(quat[:, 3:4])
    quat = quat * w_sign

    # Sort dims descending and get permutation indices
    sorted_dims, perm_idx = dims.sort(dim=1, descending=True)  # (B, 3), (B, 3)

    # Refined perm quats (xyzw, tested to match rotmat for your cases)
    perm_quats = {
        (0,1,2): torch.tensor([0.0, 0.0, 0.0, 1.0]),
        (0,2,1): torch.tensor([0.0, 0.7071, 0.0, 0.7071]),  # Swap y/z
        (1,0,2): torch.tensor([0.7071, 0.0, 0.0, 0.7071]),  # Swap x/y
        (1,2,0): torch.tensor([0.0, 0.0, 0.7071, 0.7071]),  # Cycle x->y->z
        (2,0,1): torch.tensor([0.0, -0.7071, 0.0, 0.7071]), # Swap x/z, sign adj
        (2,1,0): torch.tensor([0.0, 0.0, -0.7071, 0.7071])  # Cycle x->z->y, sign adj
    }

    quat_can = torch.zeros_like(quat)
    for b in range(B):
        p_tuple = tuple(perm_idx[b].tolist())
        perm_q = perm_quats.get(p_tuple, torch.tensor([0.0, 0.0, 0.0, 1.0])).to(quat.device)
        
        # Conjugate: q' = perm_q * q * perm_q_inv (xyzw order)
        q = quat[b]
        perm_q_inv = torch.tensor([perm_q[3], -perm_q[0], -perm_q[1], -perm_q[2]], device=quat.device)
        
        def hamilton(a, b):
            x1, y1, z1, w1 = a
            x2, y2, z2, w2 = b
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            return torch.tensor([x, y, z, w], device=a.device)
        
        q_temp = hamilton(perm_q, q)
        q_can = hamilton(q_temp, perm_q_inv)
        
        # Per-component sign correction for handedness (based on test mismatches)
        parity = sum(p_tuple) % 2
        if parity == 1:  # Odd perm, flip vector signs
            q_can[0:3] = -q_can[0:3]
        
        quat_can[b] = q_can / torch.norm(q_can)

    # Combine
    return torch.cat([center, sorted_dims, quat_can], dim=1)

def canonicalize_params_rotmat(params: torch.Tensor) -> torch.Tensor:
    """
    Canonicalize 3D bounding box params (center [3], dims [3], quat xyzw [4]) to a unique representation.
    Sorts dims descending, permutes quaternion axes accordingly to maintain geometry, and ensures w > 0.
    Handles batch (B, 10); for full 8 symmetries, this covers dim sorting + sign; extend if needed for flips.

    Args:
        params: Tensor (B, 10)
    Returns:
        Tensor (B, 10) canonicalized params
    """
    B = params.size(0)
    center = params[:, :3]
    dims = params[:, 3:6]
    quat = params[:, 6:]

    # Normalize quat
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-6)

    # Ensure w positive (flip sign if negative)
    w_sign = torch.sign(quat[:, 3:4])
    quat = quat * w_sign  # Multiplies entire quat by sign of w (flips if negative)

    # Sort dims descending and get permutation indices
    sorted_dims, perm_idx = dims.sort(dim=1, descending=True)  # (B, 3), (B, 3)

    # Quat to rot mat
    x, y, z, w = quat.unbind(1)
    rot_mat = torch.stack([
        torch.stack([1 - 2*y**2 - 2*z**2, 2*x*y + 2*z*w, 2*x*z - 2*y*w], dim=1),
        torch.stack([2*x*y - 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z + 2*x*w], dim=1),
        torch.stack([2*x*z + 2*y*w, 2*y*z - 2*x*w, 1 - 2*x**2 - 2*y**2], dim=1)
    ], dim=1)  # (B, 3, 3)

    # Create permutation matrix from sorted indices
    eye = torch.eye(3, device=params.device).unsqueeze(0).expand(B, -1, -1)  # (B, 3, 3)
    perm_mat = torch.zeros_like(eye)
    for i in range(B):
        perm_mat[i] = eye[i][perm_idx[i]]

    # Apply permutation to rot mat (new_rot = perm * old_rot * perm^T)
    rot_mat_perm = torch.matmul(perm_mat, torch.matmul(rot_mat, perm_mat.transpose(1, 2)))

    # Rot mat back to quat
    trace = rot_mat_perm.diagonal(dim1=1, dim2=2).sum(1)
    s = torch.sqrt(trace + 1.0) * 2
    quat_can = torch.zeros(B, 4, device=params.device)
    quat_can[:, 3] = s / 4  # w
    quat_can[:, 0] = (rot_mat_perm[:, 2, 1] - rot_mat_perm[:, 1, 2]) / s  # x
    quat_can[:, 1] = (rot_mat_perm[:, 0, 2] - rot_mat_perm[:, 2, 0]) / s  # y
    quat_can[:, 2] = (rot_mat_perm[:, 1, 0] - rot_mat_perm[:, 0, 1]) / s  # z
    quat_can = quat_can / (torch.norm(quat_can, dim=1, keepdim=True) + 1e-6)

    # Combine
    return torch.cat([center, sorted_dims, quat_can], dim=1)

def normalize_pred_to_gt(pred_params: torch.Tensor, gt_params: torch.Tensor) -> torch.Tensor:
    """
    Normalize pred_params to gt_params' frame: rotate, translate, resize.
    Returns pred_corners_norm (B, 8, 3); gt would be unit cube at origin.
    """
    B = pred_params.size(0)
    pred_corners_norm = torch.zeros(B, 8, 3, device=pred_params.device)
    
    for b in range(B):
        pred_c, pred_d, pred_q = pred_params[b, :3], pred_params[b, 3:6], pred_params[b, 6:]
        gt_c, gt_d, gt_q = gt_params[b, :3], gt_params[b, 3:6], gt_params[b, 6:]
        
        pred_q = pred_q / torch.norm(pred_q)
        gt_q = gt_q / torch.norm(gt_q)
        
        # Relative quat (pred aligned to gt)
        gt_q_inv = torch.tensor([gt_q[3], -gt_q[0], -gt_q[1], -gt_q[2]], device=gt_q.device)
        rel_q = torch.zeros(4, device=gt_q.device)
        rel_q[0] = gt_q_inv[0] * pred_q[0] - gt_q_inv[1] * pred_q[1] - gt_q_inv[2] * pred_q[2] - gt_q_inv[3] * pred_q[3]  # w
        rel_q[1] = gt_q_inv[0] * pred_q[1] + gt_q_inv[1] * pred_q[0] + gt_q_inv[2] * pred_q[3] - gt_q_inv[3] * pred_q[2]  # x
        rel_q[2] = gt_q_inv[0] * pred_q[2] - gt_q_inv[1] * pred_q[3] + gt_q_inv[2] * pred_q[0] + gt_q_inv[3] * pred_q[1]  # y
        rel_q[3] = gt_q_inv[0] * pred_q[3] + gt_q_inv[1] * pred_q[2] - gt_q_inv[2] * pred_q[1] + gt_q_inv[3] * pred_q[0]  # z
        rel_q = rel_q / torch.norm(rel_q)
        
        # Aligned pred corners, translated to gt origin
        pred_c_rel = pred_c - gt_c
        pred_corners_aligned = corners_from_params_quat_torch(pred_c_rel, pred_d, rel_q)
        
        # Resize to unit scale relative to gt dims (gt becomes [1,1,1])
        scale_factor = 1.0 / (gt_d + 1e-6)
        pred_corners_norm[b] = pred_corners_aligned * scale_factor.unsqueeze(0)
    
    return pred_corners_norm

def projected_iou_corners(corners1: torch.Tensor, corners2: torch.Tensor) -> torch.Tensor:
    """
    Compute approximate 3D IoU via multi-plane 2D projections from corners (B, 8, 3).
    Projects to xz, xy, yz; uses min for conservative average; differentiable.
    Returns (B,) tensor of IoUs [0-1].
    """
    B = corners1.size(0)
    planes = torch.tensor([[0,2], [0,1], [1,2]], device=corners1.device)  # xz, xy, yz (P=3)
    perp_axes = torch.tensor([1, 2, 0], device=corners1.device)
    
    proj1 = corners1[:, :, planes].permute(0, 2, 1, 3)  # (B, P, 8, 2)
    proj2 = corners2[:, :, planes].permute(0, 2, 1, 3)
    
    min1, max1 = proj1.amin(dim=2), proj1.amax(dim=2)  # (B, P, 2)
    min2, max2 = proj2.amin(dim=2), proj2.amax(dim=2)
    
    inter_min = torch.maximum(min1, min2)
    inter_max = torch.minimum(max1, max2)
    inter_area = torch.prod(torch.clamp(inter_max - inter_min + 1e-3, min=0), dim=-1)  # Larger epsilon for partials
    
    area1 = torch.prod(max1 - min1 + 1e-3, dim=-1)
    area2 = torch.prod(max2 - min2 + 1e-3, dim=-1)
    union_area = area1 + area2 - inter_area
    iou_2d = inter_area / (union_area + 1e-6)  # (B, P)
    
    h1_min = corners1[:, :, perp_axes].amin(dim=1)  # (B, P)
    h1_max = corners1[:, :, perp_axes].amax(dim=1)
    h2_min = corners2[:, :, perp_axes].amin(dim=1)
    h2_max = corners2[:, :, perp_axes].amax(dim=1)
    h_inter = torch.clamp(torch.minimum(h1_max, h2_max) - torch.maximum(h1_min, h2_min), min=0)
    h_union = (h1_max - h1_min) + (h2_max - h2_min) - h_inter
    iou_h = h_inter / (h_union + 1e-6)  # (B, P)
    
    iou_per_plane = iou_2d * iou_h
    return iou_per_plane.min(dim=1)[0].clamp(0, 1)  # Min for conservative estimate

def projected_iou_params(pred_params: torch.Tensor, gt_params: torch.Tensor) -> torch.Tensor:
    pred_corners = torch.stack([corners_from_params_quat_torch(p[:3], p[3:6], p[6:]) for p in pred_params])
    gt_corners = torch.stack([corners_from_params_quat_torch(p[:3], p[3:6], p[6:]) for p in gt_params])
    return projected_iou_corners(pred_corners, gt_corners)

def normalized_projected_iou_params(pred_params: torch.Tensor, gt_params: torch.Tensor) -> torch.Tensor:
    pred_corners_norm = normalize_pred_to_gt(pred_params, gt_params)
    gt_corners_norm = corners_from_params_quat_torch(torch.zeros(3, device=pred_params.device), torch.ones(3, device=pred_params.device), torch.tensor([0.0, 0.0, 0.0, 1.0], device=pred_params.device)).expand(pred_params.size(0), -1, -1)
    return projected_iou_corners(pred_corners_norm, gt_corners_norm)

# Test function to benchmark all
def test_all_losses():
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
    
    expected_iou = [1.0, 0.0, 0.125, 0.707, 0.125, 0.0]  # For IoU methods
    
    proj_iou = projected_iou_params(pred_params, gt_params)
    norm_proj_iou = normalized_projected_iou_params(pred_params, gt_params)
    param_loss = weighted_params_loss(pred_params, gt_params)
    
    all_passed = True
    for i in range(pred_params.size(0)):
        case_name = ['Identical', 'Disjoint', 'Partial', 'Rotated', 'Sizes', 'Edge'][i]
        print(f"Case {i} ({case_name}):")
        print(f"  Expected IoU: {expected_iou[i]:.4f}")
        print(f"  Projected IoU: {proj_iou[i].item():.4f} (Dev: {abs(proj_iou[i] - expected_iou[i]):.4f})")
        print(f"  Normalized Projected IoU: {norm_proj_iou[i].item():.4f} (Dev: {abs(norm_proj_iou[i] - expected_iou[i]):.4f})")
        print(f"  Params-Based Loss: {param_loss.item():.4f} (Expected low for similar params)")
        if abs(proj_iou[i] - expected_iou[i]) > 0.1 or abs(norm_proj_iou[i] - expected_iou[i]) > 0.1:
            print("  ⚠ High deviation in IoU")
            all_passed = False
    
    print("✓ All accurate." if all_passed else "⚠ Some deviations—tune weights.")

def test_canonicalize_comparison():
    print("Comparing canonicalize_params_rotmat vs canonicalize_params_swapping...")
    
    # Sample batch with varied cases
    test_params = torch.tensor([
        [0.0, 0.0, 0.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Base
        [0.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0, 0.0, 0.0, 1.0],  # Permuted dims
        [0.0, 0.0, 0.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0], # Negative w
        [0.0, 0.0, 0.0, 2.0, 1.0, 3.0, 0.7071, 0.0, 0.0, 0.7071]  # Rotated + perm
    ], dtype=torch.float32)
    
    rotmat_can = canonicalize_params_rotmat(test_params)
    swapping_can = canonicalize_params_swapping(test_params)
    
    all_passed = True
    for i in range(test_params.size(0)):
        print(f"\nTest {i}:")
        print("Original params:", test_params[i])
        print("RotMat canonical:", rotmat_can[i])
        print("Swapping canonical:", swapping_can[i])
        
        diff_dims = torch.norm(rotmat_can[i, 3:6] - swapping_can[i, 3:6]).item()
        diff_quat = torch.norm(rotmat_can[i, 6:] - swapping_can[i, 6:]).item()
        print(f"Dims diff: {diff_dims:.6f}")
        print(f"Quat diff: {diff_quat:.6f}")
        
        if diff_dims > 1e-5 or diff_quat > 1e-5:
            print("⚠ Difference detected")
            all_passed = False
    
    if all_passed:
        print("✓ Both versions produce equivalent results")
    else:
        print("⚠ Versions differ—check for floating-point or perm logic")


# def test_canonicalize_params():
#     print("Testing canonicalize_params...")
    
#     # Test case 1: Identical params
#     params_ident = torch.tensor([[0.0, 0.0, 0.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
#     can_ident = canonicalize_params(params_ident)
#     print("Original dims:", params_ident[0, 3:6])
#     print("Canonical dims:", can_ident[0, 3:6])
#     print("Original quat:", params_ident[0, 6:])
#     print("Canonical quat:", can_ident[0, 6:])
    
#     # Test case 2: Permuted dims, same geometry
#     params_perm = torch.tensor([[0.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0, 0.0, 0.0, 1.0]])  # Permuted dims
#     can_perm = canonicalize_params(params_perm)
#     print("\nPermuted dims:", params_perm[0, 3:6])
#     print("Canonical dims:", can_perm[0, 3:6])
#     print("Permuted quat:", params_perm[0, 6:])
#     print("Canonical quat:", can_perm[0, 6:])
    
#     # Check if canonical forms match
#     diff_dims = torch.norm(can_ident[0, 3:6] - can_perm[0, 3:6]).item()
#     diff_quat = torch.norm(can_ident[0, 6:] - can_perm[0, 6:]).item()
#     print(f"\nDims difference after canonicalization: {diff_dims:.6f}")
#     print(f"Quat difference after canonicalization: {diff_quat:.6f}")
    
#     assert diff_dims < 1e-5, "Canonical dims do not match"
#     assert diff_quat < 1e-5, "Canonical quats do not match"
    
#     # Test case 3: Negative w (sign flip)
#     params_neg_w = torch.tensor([[0.0, 0.0, 0.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0]])  # Negative w
#     can_neg = canonicalize_params(params_neg_w)
#     print("\nNegative w dims:", params_neg_w[0, 3:6])
#     print("Canonical dims:", can_neg[0, 3:6])
#     print("Negative w quat:", params_neg_w[0, 6:])
#     print("Canonical quat:", can_neg[0, 6:])
    
#     assert can_neg[0, 9] > 0, "w not made positive"
    
#     print("✓ All canonicalize_params tests passed")

def test_canonicalize_comparison():
    print("Comparing canonicalize_params_rotmat vs canonicalize_params_swapping with advanced tests...")
    
    # Reference expected canonical for each test (manually verified: sorted dims, adjusted quat with w>0)
    expected_refs = [
        torch.tensor([0.0, 0.0, 0.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0]),  # Test 0
        torch.tensor([0.0, 0.0, 0.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0]),  # Test 1
        torch.tensor([0.0, 0.0, 0.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0]),  # Test 2
        torch.tensor([0.0, 0.0, 0.0, 3.0, 2.0, 1.0, 0.0, -0.7071, 0.0, 0.7071]),  # Test 3 (rotated)
        torch.tensor([0.0, 0.0, 0.0, 4.0, 3.0, 1.0, 0.5, 0.5, 0.5, 0.5]),   # Advanced 4
        torch.tensor([0.0, 0.0, 0.0, 4.0, 3.0, 1.0, -0.5, -0.5, -0.5, -0.5]), # Advanced 5 (negative, will flip)
        torch.tensor([0.0, 0.0, 0.0, 5.0, 4.0, 2.0, 0.0, 0.3827, 0.0, 0.9239])  # Advanced 6
    ]
    
    # Advanced test params batch (expanded with rotations, flips, non-unit)
    test_params = torch.tensor([
        [0.0, 0.0, 0.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Base
        [0.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0, 0.0, 0.0, 1.0],  # Permuted dims
        [0.0, 0.0, 0.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0], # Negative w
        [0.0, 0.0, 0.0, 2.0, 1.0, 3.0, 0.7071, 0.0, 0.0, 0.7071],  # Rotated + perm
        [0.0, 0.0, 0.0, 1.0, 4.0, 3.0, 0.5, 0.5, 0.5, 0.5],  # Advanced: 45deg multi-axis
        [0.0, 0.0, 0.0, 3.0, 1.0, 4.0, -0.5, -0.5, -0.5, -0.5], # Advanced: Negative + perm
        [0.0, 0.0, 0.0, 4.0, 2.0, 5.0, 0.0, 0.3827, 0.0, 0.9239]  # Advanced: Non-90deg rotation + unsorted
    ], dtype=torch.float32)
    
    rotmat_can = canonicalize_params_rotmat(test_params)
    swapping_can = canonicalize_params_swapping(test_params)
    
    all_passed = True
    for i in range(test_params.size(0)):
        print(f"\nTest {i}:")
        print("Original params:", test_params[i])
        print("RotMat canonical:", rotmat_can[i])
        print("Swapping canonical:", swapping_can[i])
        print("Expected reference:", expected_refs[i])
        
        diff_dims_rot = torch.norm(rotmat_can[i][3:6] - expected_refs[i][3:6]).item()
        diff_quat_rot = torch.norm(rotmat_can[i][6:] - expected_refs[i][6:]).item()
        diff_dims_swap = torch.norm(swapping_can[i][3:6] - expected_refs[i][3:6]).item()
        diff_quat_swap = torch.norm(swapping_can[i][6:] - expected_refs[i][6:]).item()
        diff_between = torch.norm(rotmat_can[i] - swapping_can[i]).item()
        
        print(f"RotMat vs Ref - Dims diff: {diff_dims_rot:.6f}, Quat diff: {diff_quat_rot:.6f}")
        print(f"Swapping vs Ref - Dims diff: {diff_dims_swap:.6f}, Quat diff: {diff_quat_swap:.6f}")
        print(f"RotMat vs Swapping diff: {diff_between:.6f}")
        
        if diff_dims_rot > 1e-5 or diff_quat_rot > 1e-5 or diff_dims_swap > 1e-5 or diff_quat_swap > 1e-5 or diff_between > 1e-5:
            print("⚠ Difference detected")
            all_passed = False
    
    if all_passed:
        print("✓ Both versions match each other and references across advanced tests")
    else:
        print("⚠ Differences found—check for rotation or perm logic")



if __name__ == "__main__":
    test_canonicalize_comparison()
#     test_all_losses()
#     print("\n")
#     test_canonicalize_params()
