"""
Vision classifier using ConvNeXt-Base backbone with a linear head for Stage 0 pretraining.
"""

import torch
import torch.nn as nn
from model import ConvNeXtImageEncoder, GeM


class VisionFoodClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.image_encoder = ConvNeXtImageEncoder(pretrained=pretrained)
        self.pooling = GeM()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.image_encoder(x)
        feat = self.pooling(feat_map).flatten(1)
        return self.head(feat)

    def extract_backbone_state_dict(self):
        """Returns the state_dict for image_encoder (ConvNeXt)."""
        return self.image_encoder.state_dict()
