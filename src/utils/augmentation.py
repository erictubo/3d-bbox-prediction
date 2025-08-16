"""
Data Augmentation Utilities

This module provides augmentation functions for 3D bounding box datasets.
It handles both 2D image augmentations and 3D geometric transformations while maintaining
consistency between different data modalities (RGB images, point clouds, bounding boxes).

Key Functions:
- crop_and_resize_rgb: Mask-based RGB cropping with aspect ratio preservation
- apply_mask_context
- apply_flip: Consistent horizontal/vertical flipping
- apply_xy_translation: 2D translation of points and bounding boxes
- apply_z_rotation: 3D rotation around Z-axis
- apply_scale: Uniform scaling of points and bounding boxes

Testing in test_augmentations.py
"""

import torch
import numpy as np
from PIL import Image
import albumentations as A
import torchvision.transforms as T
from scipy.ndimage import binary_dilation


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

def apply_mask_context(mask, context_size=5):
    structure = np.ones((context_size, context_size), dtype=bool)
    mask = binary_dilation(mask > 0, structure=structure).astype(mask.dtype)
    return mask

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
    center_np = center.numpy()  # Convert to NumPy for bbox

    if flip_type == 'horizontal':
        # Image/Mask: Horizontal flip
        aug = A.Compose([A.HorizontalFlip(p=1.0)])
        augmented = aug(image=np.array(image))
        flipped_image = Image.fromarray(augmented['image'])

        # Flip points and bbox corners (negate x relative to center_x)
        points[:, 0] = 2 * center[0] - points[:, 0]
        bbox_corners[:, 0] = 2 * center_np - bbox_corners[:, 0]

    elif flip_type == 'vertical':
        # Image/Mask: Vertical flip
        aug = A.Compose([A.VerticalFlip(p=1.0)])
        augmented = aug(image=np.array(image))
        flipped_image = Image.fromarray(augmented['image'])

        # Flip points and bbox corners (negate y relative to center_y)
        points[:, 1] = 2 * center[1] - points[:, 1]
        bbox_corners[:, 1] = 2 * center_np[1] - bbox_corners[:, 1]

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

def apply_scale(points: torch.Tensor, bbox_params: torch.Tensor, factor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scale points and bbox params (center + dims) by factor.
    - factor: float, e.g., random.uniform(0.8, 1.2)
    Returns updated points, bbox_params.
    """
    scaled_points = points * factor
    scaled_params = bbox_params.clone()
    scaled_params[:6] *= factor  # Scale center (0:3) and dims (3:6); quat (6:) unchanged
    return scaled_points, scaled_params
