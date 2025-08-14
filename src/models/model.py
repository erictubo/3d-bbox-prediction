import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights  # Pre-trained RGB backbone

class PointNetBackbone(nn.Module):
    """Simple PointNet for point cloud feature extraction."""
    def __init__(self, out_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, out_dim), nn.ReLU()
        )

    def forward(self, points):  # points: (B, M, 3)
        feat = self.mlp(points)  # (B, M, out_dim)
        return torch.max(feat, dim=1)[0]  # Global max pooling: (B, out_dim)

class ThreeDBBoxModel(nn.Module):
    def __init__(self):
        super().__init__()
        # RGB backbone: ResNet-18, modified to output 512-dim features
        self.rgb_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.rgb_backbone.fc = nn.Linear(self.rgb_backbone.fc.in_features, 512)
        
        # Point cloud backbone
        self.pc_backbone = PointNetBackbone(512)
        
        # Fusion layer: Combine 512 (RGB) + 512 (PC) -> 256
        self.fusion = nn.Linear(512 + 512, 256)
        self.dropout = nn.Dropout(0.5)
        
        # Regression head: Output 10 params (center[3], dims[3], quat[4])
        self.head = nn.Linear(256, 10)

    def forward(self, rgb_crops, points_list):
        # rgb_crops: (B, 3, 224, 224) - batched per-object RGB crops
        # points_list: (B, M, 3) - batched per-object masked points
        
        rgb_feat = self.rgb_backbone(rgb_crops)  # (B, 512)
        pc_feat = self.pc_backbone(points_list)  # (B, 512)
        
        fused = torch.relu(self.fusion(torch.cat([rgb_feat, pc_feat], dim=1)))  # (B, 256)
        fused = self.dropout(fused)
        
        raw_preds = self.head(fused)  # (B, 10)
        
        # Normalize quaternion part for valid rotation
        quat = raw_preds[:, 6:10]  # (B, 4)
        quat = quat / torch.norm(quat, dim=1, keepdim=True)  # Unit norm (B, 4)
        
        # Combine back with center and dims
        preds = torch.cat([raw_preds[:, :6], quat], dim=1)  # (B, 10)
        
        return preds

