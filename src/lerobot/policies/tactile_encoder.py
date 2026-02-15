"""Shared tactile encoder modules for all policies.

These encoders transform 2D tactile sensor data (H, W) into a 1D feature
vector of configurable dimension. Two architectures are provided:

- TactileCNN: A simple 3-layer conv net with BatchNorm, ReLU, MaxPool,
  then a 2-layer MLP to produce the output embedding.
- TactileAttentionCNN: A 3-layer conv net with a spatial attention
  mechanism (1x1 conv -> sigmoid) and dual global pooling (avg + max),
  then a 2-layer MLP.
"""

import torch
import torch.nn.functional as F
from torch import nn


class TactileCNN(nn.Module):
    """Tactile CNN encoder for extracting features from 2D tactile arrays."""

    def __init__(self, input_shape=(16, 32), feature_dim=256, dropout=0.3):
        """
        Args:
            input_shape: tuple (height, width) of tactile sensor
            feature_dim: output feature dimension for transformer
            dropout: dropout probability
        """
        super(TactileCNN, self).__init__()

        self.input_shape = input_shape
        self.feature_dim = feature_dim

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        # Calculate the size after convolutions and pooling
        # After 3 pooling layers: H/8 x W/8
        feature_h = input_shape[0] // 8
        feature_w = input_shape[1] // 8
        conv_output_dim = 128 * feature_h * feature_w

        # Fully connected layers to get desired feature dimension
        self.fc1 = nn.Linear(conv_output_dim, 512)
        self.fc2 = nn.Linear(512, feature_dim)

    def forward(self, x):
        # Add channel dimension if not present
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class TactileAttentionCNN(nn.Module):
    """Tactile encoder with spatial attention and dual pooling."""

    def __init__(self, input_shape=(16, 32), feature_dim=256, dropout=0.4):
        """
        Args:
            input_shape: tuple (height, width) of tactile sensor
            feature_dim: output feature dimension for transformer
            dropout: dropout probability
        """
        super(TactileAttentionCNN, self).__init__()

        self.input_shape = input_shape
        self.feature_dim = feature_dim

        # Feature extraction
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        # Spatial attention
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Feature projection to desired dimension
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Spatial attention
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Dual pooling
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)

        # Concatenate both pooling results
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = x.view(x.size(0), -1)

        # Feature projection
        x = self.fc(x)

        return x


def build_tactile_encoder(
    encoder_type: str = "cnn",
    input_shape: tuple[int, int] = (16, 32),
    feature_dim: int = 256,
    dropout: float = 0.3,
) -> nn.Module:
    """Factory to create a tactile encoder.

    Args:
        encoder_type: "cnn" or "attention"
        input_shape: (H, W) of tactile sensor
        feature_dim: output feature dimension
        dropout: dropout probability

    Returns:
        An nn.Module that maps (B, H, W) or (B, 1, H, W) -> (B, feature_dim)
    """
    if encoder_type == "cnn":
        return TactileCNN(input_shape=input_shape, feature_dim=feature_dim, dropout=dropout)
    elif encoder_type == "attention":
        return TactileAttentionCNN(input_shape=input_shape, feature_dim=feature_dim, dropout=dropout)
    else:
        raise ValueError(
            f"Unknown tactile encoder type: {encoder_type}. Expected 'cnn' or 'attention'."
        )
