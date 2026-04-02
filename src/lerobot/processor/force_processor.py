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
"""Force vector processing steps."""

from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ObservationProcessorStep
from lerobot.utils.constants import OBS_FORCE_VEC


class ForceValidationProcessorStep(ObservationProcessorStep):
    """Validate force vector data format and dimensions."""

    def __init__(self, expected_dim: int = 6):
        self.expected_dim = expected_dim

    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        for key in list(obs.keys()):
            if key != OBS_FORCE_VEC and not key.startswith(OBS_FORCE_VEC + "."):
                continue
            force_vec = obs[key]

            if isinstance(force_vec, np.ndarray):
                force_vec = torch.from_numpy(force_vec).float()

            if force_vec.dim() == 1:
                actual_dim = force_vec.shape[0]
            elif force_vec.dim() == 2:
                actual_dim = force_vec.shape[-1]
            else:
                raise ValueError(
                    f"Expected 1D ({self.expected_dim},) or 2D (B, {self.expected_dim}) tensor for '{key}', "
                    f"but got {force_vec.dim()}D tensor."
                )

            if actual_dim != self.expected_dim:
                raise ValueError(
                    f"Force vector shape mismatch for '{key}'. Expected trailing dim {self.expected_dim}, "
                    f"got {actual_dim}."
                )

            obs[key] = force_vec

        return obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
