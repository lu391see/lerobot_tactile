#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tactile sensor data processing steps"""

import torch
import numpy as np
from typing import Any

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ObservationProcessorStep
from lerobot.utils.constants import OBS_TACTILE


class TactileNormalizationProcessorStep(ObservationProcessorStep):
    """Normalize 3-axis tactile sensor data (X, Y, Z forces)"""

    def __init__(self, force_max=0.4):
        """
        Args:
            force_max: Expected maximum force magnitude in Newtons to scale data to [-1, 1]
        """
        self.force_max = force_max

    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        for key in list(obs.keys()):
            if key != OBS_TACTILE and not key.startswith(OBS_TACTILE + "."):
                continue
            tactile_data = obs[key]

            # Convert to tensor if numpy array
            if isinstance(tactile_data, np.ndarray):
                tactile_data = torch.from_numpy(tactile_data).float()

            # Clamp between -1 and 1 to prevent OOD spikes from blowing up the Transformer
            tactile_data = torch.clamp(tactile_data, min=-self.force_max, max=self.force_max) / self.force_max

            obs[key] = tactile_data

        return obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Features remain unchanged after normalization"""
        return features


class TactileValidationProcessorStep(ObservationProcessorStep):
    """Validate tactile sensor data format and 3D dimensions"""

    # Default changed to (Channels, Height, Width)
    def __init__(self, expected_shape=(3, 40, 40)):
        """
        Args:
            expected_shape: Expected shape of tactile sensor array (C, H, W)
        """
        self.expected_shape = tuple(expected_shape)

    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        for key in list(obs.keys()):
            if key != OBS_TACTILE and not key.startswith(OBS_TACTILE + "."):
                continue
            tactile_data = obs[key]

            # Convert to tensor if needed
            if isinstance(tactile_data, np.ndarray):
                tactile_data = torch.from_numpy(tactile_data).float()

            # Check dimensions (We explicitly require the channel dimension now)
            if tactile_data.dim() == 3:
                # (C, H, W) format - single sample
                pass
            elif tactile_data.dim() == 4:
                # (B, C, H, W) format - batched
                pass
            else:
                raise ValueError(
                    f"Expected 3D (C, H, W) or 4D (B, C, H, W) tensor for '{key}', "
                    f"but got {tactile_data.dim()}D tensor."
                )

            # Validate shape (Check the last 3 dimensions to handle both batched and unbatched)
            actual_shape = tuple(tactile_data.shape[-3:])
            if actual_shape != self.expected_shape:
                raise ValueError(
                    f"Tactile data shape mismatch for '{key}'. Expected {self.expected_shape}, "
                    f"got {actual_shape}"
                )

            obs[key] = tactile_data

        return obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Features remain unchanged after validation"""
        return features