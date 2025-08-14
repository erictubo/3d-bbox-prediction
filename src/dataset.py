import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as T
import albumentations as A
import random

from src.utils.data_utils import crop_and_resize_rgb, apply_flip, apply_z_rotation, apply_xy_translation, params_from_corners_quat

# TODO:
# - rotation - fix + probably before crop
# - flipping

class BBoxDataset(Dataset):
    def __init__(self, data_paths, max_points=1024, crop_size=(224, 224), aug_prob=0.5):
        """
        data_paths: list of dicts, each {'image': str, 'mask': str, 'pc': str, 'bbox': str}
        aug_prob: Probability to apply augmentations (0.5 default).
        """
        self.data_paths = data_paths
        self.max_points = max_points
        self.crop_size = crop_size
        self.aug_prob = aug_prob
        
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # Albumentations for RGB (applied to crops)
        self.rgb_aug = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)  # Added for variance
        ], p=1.0)  # Apply with probability 1, but internals have probs

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        item = self.data_paths[idx]

        # Load data
        rgb = Image.open(item['image']).convert('RGB')
        masks = np.load(item['mask'])  # (N, H, W)
        pc = np.load(item['pc'])  # (3, H, W) -> reshape to (H*W, 3)
        pc = pc.transpose(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
        gt_bboxes = np.load(item['bbox'])  # (N, 8, 3)

        rgb_crops = []
        points_list = []
        targets = []

        for i in range(len(masks)):
            mask = masks[i]  # (H, W)
            gt_bbox_corners = gt_bboxes[i]  # (8, 3)

            # Extract masked points (filter pc where mask > 0)
            masked_idx = np.where(mask.flatten() > 0)[0]
            masked_points = pc[masked_idx]  # (M, 3)
            if len(masked_points) > self.max_points:
                masked_points = masked_points[np.random.choice(len(masked_points), self.max_points, replace=False)]
            elif len(masked_points) < self.max_points:
                padded = np.zeros((self.max_points, 3))
                padded[:len(masked_points)] = masked_points
                masked_points = padded

            # Crop and resize RGB using mask bbox
            crop = crop_and_resize_rgb(rgb, mask, padding=0.1, crop_size=self.crop_size)  # Returns PIL

            masked_points_t = torch.from_numpy(masked_points).float()

            # Apply flip
            if random.random() < 0.5:
                flip_type = random.choice(['horizontal', 'vertical'])
                crop, masked_points_t, gt_bbox_corners = apply_flip(
                    crop, masked_points_t, gt_bbox_corners, flip_type
                )

            # Convert GT bbox to params (center, dims, quat)
            target_params_t = torch.from_numpy(params_from_corners_quat(gt_bbox_corners)).float()  # (10,)

            # Apply augmentations with probability
            if random.random() < self.aug_prob:

                # X/Y Translation
                shift_x = random.uniform(-0.15, 0.15)
                shift_y = random.uniform(-0.15, 0.15)
                masked_points_t, target_params_t = apply_xy_translation(masked_points_t, target_params_t, shift_x, shift_y)
                
                # # Z-Rotation - doesn't make sense because the bounding box does not update
                # angle_z = random.uniform(-30, 30)  # Degrees
                # crop, masked_points_t, target_params_t = apply_z_rotation(crop, masked_points_t, target_params_t, angle_z)

            # Apply RGB augmentations (non-geometric)
            crop_np = np.array(crop)
            augmented = self.rgb_aug(image=crop_np)
            crop = Image.fromarray(augmented['image'])
            crop = self.transform(crop)  # Apply torchvision transforms

            rgb_crops.append(crop)
            points_list.append(masked_points_t)
            targets.append(target_params_t)

        # print(f"Item {idx}: rgb_crops shape: {torch.stack(rgb_crops).shape}, points_list shape: {torch.stack(points_list).shape}, targets shape: {torch.stack(targets).shape}")

        return {
            'rgb_crops': torch.stack(rgb_crops),  # (N, 3, 224, 224)
            'points_list': torch.stack(points_list),  # (N, max_points, 3)
            'targets': torch.stack(targets)  # (N, 10)
        }

# Custom collate to flatten per-scene lists into batches (unchanged)
def flatten_collate(batch):
    flat_rgb = torch.cat([item['rgb_crops'] for item in batch])
    flat_points = torch.cat([item['points_list'] for item in batch])
    flat_targets = torch.cat([item['targets'] for item in batch])
    return {'rgb_crops': flat_rgb, 'points': flat_points, 'targets': flat_targets}
