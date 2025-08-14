import torch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import albumentations as A
import torchvision.transforms as T
import open3d as o3d

# - AUGMENTATION
# - BBOX CONVERSION
# - TESTING


# ==================== AUGMENTATION ====================

def crop_and_resize_rgb(rgb: Image.Image, mask: np.ndarray, padding: float = 0.1,
                        crop_size: tuple[int, int] = (224, 224)) -> tuple[Image.Image, np.ndarray]:
    """
    Crop RGB (PIL Image) to mask's bounding box with padding.
    Resize with black background to preserve aspect ratio.
    padding: Fraction to expand bbox (e.g., 0.2 = 20%).
    mask: (H, W) numpy array.
    """
    # Find bounding box from mask
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return T.ToTensor()(rgb.resize(crop_size))  # Fallback
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    if padding > 0.0:
        height, width = max_row - min_row, max_col - min_col
        margin_h = int(height * padding)
        margin_w = int(width * padding)
        min_row = max(0, min_row - margin_h)
        max_row = min(rgb.height, max_row + margin_h)
        min_col = max(0, min_col - margin_w)
        max_col = min(rgb.width, max_col + margin_w)

    # Crop
    crop = rgb.crop((min_col, min_row, max_col, max_row))

    # Pad with black to preserve aspect (letterbox)
    crop_w, crop_h = crop.size
    if crop_w == 0 or crop_h == 0:
        return Image.new('RGB', crop_size, (0, 0, 0))
    
    ratio = min(crop_size[0] / crop_w, crop_size[1] / crop_h)
    new_w, new_h = int(crop_w * ratio), int(crop_h * ratio)
    resized_crop = crop.resize((new_w, new_h), Image.BILINEAR)

    padded = Image.new('RGB', crop_size, (0, 0, 0))  # Black background
    pad_left = (crop_size[0] - new_w) // 2
    pad_top = (crop_size[1] - new_h) // 2
    padded.paste(resized_crop, (pad_left, pad_top))

    return padded  # Return PIL for further transforms in Dataset


def apply_flip(image: Image.Image, points: torch.Tensor, bbox_corners: np.ndarray, flip_type: str = 'horizontal'):
    """
    Apply consistent flip to all items using centroid as flip center.
    - flip_type: 'horizontal' or 'vertical'.
    - Image/Mask: 2D flip.
    - Points/BBox Corners: Flip coords relative to centroid (middle point).
    - Returns updated image, mask, points, and new bbox_params (from flipped corners).
    """
    # Compute flip center (use points mean for robustness; fallback to bbox if needed)
    if len(points) > 0:
        center = points.mean(dim=0)  # (3,)
    else:
        center = torch.from_numpy(bbox_corners).float().mean(dim=0)  # (3,)

    if flip_type == 'horizontal':
        # Image/Mask: Horizontal flip
        aug = A.Compose([A.HorizontalFlip(p=1.0)])
        augmented = aug(image=np.array(image))
        flipped_image = Image.fromarray(augmented['image'])

        # Flip points and bbox corners (negate x relative to center_x)
        points[:, 0] = 2 * center[0] - points[:, 0]
        bbox_corners[:, 0] = 2 * center[0] - bbox_corners[:, 0]  # Assuming bbox_corners (8,3) numpy

    elif flip_type == 'vertical':
        # Image/Mask: Vertical flip
        aug = A.Compose([A.VerticalFlip(p=1.0)])
        augmented = aug(image=np.array(image))
        flipped_image = Image.fromarray(augmented['image'])

        # Flip points and bbox corners (negate y relative to center_y)
        points[:, 1] = 2 * center[1] - points[:, 1]
        bbox_corners[:, 1] = 2 * center[1] - bbox_corners[:, 1]

    else:
        raise ValueError("Invalid flip_type")

    return flipped_image, points, bbox_corners


# Hamilton product for quaternion composition (for rotations)
def hamilton_product(q1, q2):
    """Compose two quaternions (x,y,z,w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return torch.tensor([x, y, z, w])

def apply_xy_translation(points: torch.Tensor, bbox_params: torch.Tensor, shift_x: float, shift_y: float):
    """
    Apply small X/Y translation to points and bbox params (center shifted).
    Image/mask unchanged (translations don't affect 2D crop visibly without reprojection).
    Returns updated points, bbox_params.
    """
    shift = torch.tensor([shift_x, shift_y, 0.0], dtype=torch.float)  # Z unchanged

    # Translate points
    translated_points = points + shift

    # Translate bbox center (dims/quat unchanged)
    center = bbox_params[:3]
    dims = bbox_params[3:6]
    quat = bbox_params[6:]
    translated_center = center + shift
    translated_params = torch.cat([translated_center, dims, quat])

    return translated_points, translated_params

def apply_z_rotation(image: Image.Image, points: torch.Tensor, bbox_params: torch.Tensor, angle_deg: float):
    """
    Apply Z-rotation (yaw) to all items consistently.
    - Points: 3D rotation matrix.
    - BBox Params: Rotate center and compose quat.
    Returns updated image, mask, points, bbox_params.
    """
    angle_rad = torch.deg2rad(torch.tensor(angle_deg))

    # Convert image to numpy
    image_np = np.array(image)

    # Apply rotation with reflect mode to avoid size issues
    aug = A.Compose([A.Rotate(limit=(angle_deg, angle_deg), p=1.0, border_mode='constant')])
    augmented = aug(image=image_np)
    rotated_image = Image.fromarray(augmented['image'])

    # 3D rotation for points and bbox (unchanged, as it's MPS-safe)
    cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
    rot_mat = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=torch.float)

    rotated_points = torch.matmul(points, rot_mat.T)

    center = bbox_params[:3]
    dims = bbox_params[3:6]
    quat = bbox_params[6:]
    rotated_center = torch.matmul(center, rot_mat.T)
    rot_quat = torch.tensor([0.0, 0.0, torch.sin(angle_rad / 2), torch.cos(angle_rad / 2)])
    rotated_quat = hamilton_product(rot_quat, quat)
    norm = torch.norm(rotated_quat)
    if norm < 1e-6:
        rotated_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    else:
        rotated_quat = rotated_quat / norm
    rotated_params = torch.cat([rotated_center, dims, rotated_quat])

    return rotated_image, rotated_points, rotated_params


# ==================== BBOX CONVERSION ====================

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

# def params_from_corners_6d(corners):
#     """
#     Convert 8 corners (8, 3) to center (3), dims (3), rot_6d (6).
#     Uses Open3D for bbox, then extracts 6D from rotation matrix.
#     """
#     # Use Open3D for center, dims, rot_mat (from previous impl)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(corners)
#     bbox = pcd.get_oriented_bounding_box()
#     center = np.asarray(bbox.center)
#     dims = np.asarray(bbox.extent)
#     rot_mat = np.asarray(bbox.R)
    
#     # Extract first two columns as 6D
#     rot_6d = rot_mat[:, :2].flatten()  # (6,)
#     return np.concatenate([center, dims, rot_6d])


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


# ==================== TESTING ====================

def test_round_trip_numpy(corners_np, tolerance=1e-2, mode='full'):  # **Added mode param for flexibility**
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
    
def test_round_trip_torch(corners_np, tolerance=1e-2, mode='full'):
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


# def test_round_trip_6d(corners_np, tolerance=1e-2):
#     params = params_from_corners_6d(corners_np)
#     center = torch.from_numpy(params[:3]).float()
#     dims = torch.from_numpy(params[3:6]).float()
#     rot_6d = torch.from_numpy(params[6:]).float()
#     reconstructed = corners_from_params_6d(center, dims, rot_6d).numpy()
    
#     # Mean min distance
#     dists = cdist(corners_np, reconstructed)
#     min_dists = np.min(dists, axis=1)
#     mean_dist = np.mean(min_dists)
    
#     if mean_dist <= tolerance:
#         print(f"6D round-trip successful! Mean dist: {mean_dist}")
#         return True
#     else:
#         print(f"6D round-trip failed! Mean dist: {mean_dist}")
#         return False