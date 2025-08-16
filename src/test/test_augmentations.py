"""
Testing of Augmenations
- augmentation.py

1. Mask context
2. Flip (horizontal & vertical)
3. XY translation
4. Z rotation
5. Scale
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import numpy as np
import torch
from PIL import Image

from src.utils.augmentation import apply_mask_context, apply_flip, apply_xy_translation, apply_z_rotation, apply_scale
from src.utils.bbox_conversion import params_from_corners_quat, corners_from_params_torch
from src.utils.visualization import visualize_raw_data_objects


def test_mask_context():
    """Test mask dilation expands area correctly."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:6, 4:6] = True  # Small square
    dilated = apply_mask_context(mask)
    assert dilated.sum() > mask.sum(), "Dilation failed to expand mask"
    # Dummy pc for filtering test
    pc = np.random.rand(100, 3)  # Flattened (H*W=100)
    masked_idx = np.where(dilated.flatten() > 0)[0]
    assert len(masked_idx) > 4, "Expanded mask should filter more points"  # Original ~4
    print("Mask context test passed")

def test_flip():
    """Test flip consistency across image, points, bbox."""
    rgb = Image.new('RGB', (4, 4), 'white')
    mask = np.ones((4, 4), dtype=bool)
    points = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    bbox_corners = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]])[np.newaxis, :, :]  # (1, 8, 3)
    orig_center = points.mean(0)
    # orig_center_x = orig_center[0]  # Extract x-coordinate only
    # orig_center_y = orig_center[1]  # Extract y-coordinate only

    # save before flip
    visualize_raw_data_objects(rgb, mask, points, bbox_corners, save_dir='vis/aug', name='before')
    
    flipped_img, flipped_points, flipped_bbox = apply_flip(rgb, points.clone(), bbox_corners[0].copy(), 'horizontal')
    assert flipped_img.size == rgb.size, "Image size changed after flip"
    # assert torch.allclose(flipped_points[:, 0], 2 * orig_center_x - points[:, 0]), "Points x not flipped"
    # assert np.allclose(flipped_bbox[:, 0], 2 * orig_center_x - bbox_corners[:, 0]), "BBox x not flipped"

    # save after flip
    visualize_raw_data_objects(flipped_img, mask, points, flipped_bbox[np.newaxis, :, :], save_dir='vis/aug', name='h_flip')
    
    # Consistency: Params from flipped bbox should differ appropriately
    orig_params = params_from_corners_quat(bbox_corners[0])
    flipped_params = params_from_corners_quat(flipped_bbox)
    assert not np.allclose(orig_params, flipped_params), "Params unchanged after flip"

    flipped_img, flipped_points, flipped_bbox = apply_flip(rgb, points.clone(), bbox_corners[0].copy(), 'vertical')
    assert flipped_img.size == rgb.size, "Image size changed after flip"
    # assert torch.allclose(flipped_points[:, 1], 2 * orig_center_y - points[:, 1]), "Points y not flipped"
    # assert np.allclose(flipped_bbox[:, 1], 2 * orig_center_y - bbox_corners[:, 1]), "BBox y not flipped"
    # assert not np.allclose(orig_params, flipped_params), "Params unchanged after flip"

    # save after flip
    visualize_raw_data_objects(flipped_img, mask, points, flipped_bbox[np.newaxis, :, :], save_dir='vis/aug', name='v_flip')
    # print("Flip test passed")

def test_xy_translation():
    """Test translation shifts points and bbox params consistently."""
    points = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    bbox_params = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0])
    shift_x, shift_y = 0.5, -0.5
    
    translated_points, translated_params = apply_xy_translation(points.clone(), bbox_params.clone(), shift_x, shift_y)
    assert torch.allclose(translated_points, points + torch.tensor([shift_x, shift_y, 0.0])), "Points not translated"
    assert torch.allclose(translated_params[:3], bbox_params[:3] + torch.tensor([shift_x, shift_y, 0.0])), "Center not translated"
    assert torch.allclose(translated_params[3:], bbox_params[3:]), "Dims/quat changed unexpectedly"
    
    # Consistency: Corners from translated params should match translated points' shift
    orig_corners = corners_from_params_torch(bbox_params[:3], bbox_params[3:6], bbox_params[6:])
    trans_corners = corners_from_params_torch(translated_params[:3], translated_params[3:6], translated_params[6:])
    assert torch.allclose(trans_corners - orig_corners, torch.tensor([shift_x, shift_y, 0.0])), "Corners not shifted consistently"
    print("XY translation test passed")

def test_z_rotation():
    """Test z rotation consistency across image, points, params."""
    rgb = Image.new('RGB', (4, 4), 'white')
    points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    bbox_params = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    angle = 90.0
    
    rotated_img, rotated_points, rotated_params = apply_z_rotation(rgb, points.clone(), bbox_params.clone(), angle)
    assert rotated_img.size == rgb.size, "Image size changed after rotation"
    assert torch.allclose(rotated_points, torch.tensor([0.0, 1.0, 0.0]), atol=1e-4), "Point not rotated 90 degrees"
    assert torch.allclose(rotated_params[3:6], bbox_params[3:6]), "Dims changed after rotation"
    assert not torch.allclose(rotated_params[6:], bbox_params[6:]), "Quat unchanged after rotation"
    
    # Consistency: Rotated params should produce rotated corners
    orig_corners = corners_from_params_torch(bbox_params[:3], bbox_params[3:6], bbox_params[6:])
    rot_corners = corners_from_params_torch(rotated_params[:3], rotated_params[3:6], rotated_params[6:])
    assert not torch.allclose(rot_corners, orig_corners), "Corners unchanged after rotation"
    print("Z rotation test passed")

def test_scale():
    """Test scale augmentation (new implementation suggestion)."""
    # Assuming a new apply_scale function (see below)
    points = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    bbox_params = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    factor = 1.5
    
    scaled_points, scaled_params = apply_scale(points.clone(), bbox_params.clone(), factor)
    assert torch.allclose(scaled_points, points * factor), "Points not scaled"
    assert torch.allclose(scaled_params[3:6], bbox_params[3:6] * factor), "Dims not scaled"
    assert torch.allclose(scaled_params[:3], bbox_params[:3] * factor), "Center scaled (adjust if not desired)"
    print("Scale test passed")


if __name__ == '__main__':
    # Run all tests
    test_mask_context()
    test_flip()
    test_xy_translation()
    test_z_rotation()
    test_scale()
    print("All tests completed")
