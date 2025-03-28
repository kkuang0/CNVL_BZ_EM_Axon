import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class MultiTaskEfficientNet(nn.Module):
    def __init__(self):
        super(MultiTaskEfficientNet, self).__init__()
        
        # Load a pretrained EfficientNetV2-B2 
        self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)

        # Replace final classifier with an identity
        # so we can attach multiple heads ourselves
        self.backbone.classifier = nn.Identity()

        # EfficientNetV2-B2's last feature dimension is typically 1408
        in_features = 1408
        
        # Create multi-task heads. Feel free to adjust hidden dim, dropout, etc.
        self.pathology_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # ASD vs CTR
        )
        
        self.region_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)  # A25, A46, OFC
        )
        
        self.depth_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # DWM vs SWM
        )
        
        # Modify first convolution to accept 1-channel input instead of 3
        # The first conv is typically self.backbone.features[0][0]
        first_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=(first_conv.bias is not None)
        )
        # Copy mean of existing weights to the new conv
        with torch.no_grad():
            new_conv.weight = nn.Parameter(torch.mean(first_conv.weight, dim=1, keepdim=True))
        if first_conv.bias is not None:
            new_conv.bias = first_conv.bias
        
        self.backbone.features[0][0] = new_conv
        
    def forward(self, x):
        # Extract features using the EfficientNet backbone
        x = self.backbone(x)
        
        # Pass the pooled features to each classification head
        pathology_pred = self.pathology_head(x)
        region_pred = self.region_head(x)
        depth_pred = self.depth_head(x)
        
        return (pathology_pred, region_pred, depth_pred)
