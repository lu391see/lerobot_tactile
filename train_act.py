#!/usr/bin/env python3
"""
Training script
"""
# adapted from lerobot examples and lefranx

from pathlib import Path
import os

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig, ImageTransformsConfig, WandBConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.transforms import ImageTransformConfig
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train

WANDB_PROJECT = "insert-pinch-act-v3"
TRAIN_NAME = "wrist-black"

REPO_NAME = "insert-pinch-v3"
SEED = 42
BLACKOUT_CAMERAS = True

suffix_input_filter = [
    "images.thumb-tip",
    "images.index-tip",
    "tactile.thumb",
    "tactile.index",
    # "images.wrist",
]


def main():
    """Run training with manually constructed config to bypass draccus issues."""

    HF_LEROBOT_HOME = os.getenv("HF_LEROBOT_HOME", f"{Path.home()}/.cache/lerobot")
    HF_USER = os.getenv("HF_USER", "lerobot")

    output_directory = Path(f"{HF_LEROBOT_HOME}/outputs/train/{TRAIN_NAME}{SEED}")
    dataset_directoy = f"{HF_LEROBOT_HOME}/data/{REPO_NAME}"

    dataset_metadata = LeRobotDatasetMetadata(dataset_directoy)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    # filter input features
    input_features = {
        key: ft for key, ft in input_features.items() if not any(key.endswith(suffix) for suffix in suffix_input_filter)
    }
    print("input features:", list(input_features.keys()))
    print("output features:", list(output_features.keys()))

    if BLACKOUT_CAMERAS:
        # Add camera blackout transform to input features
        selected_tf = {
            "camera_blackout": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.0, 0.0)},
            )
        }
    else:
        selected_tf = {}

    # Use the local dataset instead of trying to download from hub
    dataset_config = DatasetConfig(
        repo_id=f"{HF_LEROBOT_HOME}/{REPO_NAME}",  # Absolute local path
        image_transforms=ImageTransformsConfig(
            enable=True,
            max_num_transforms=1,
            random_order=False,
            tfs=selected_tf,
        ),
    )

    # Create ACT policy config
    policy_config = ACTConfig(
        device="cuda",  # should be chosen automatically if available
        input_features=input_features,
        output_features=output_features,
        chunk_size=30,
        n_action_steps=30,
        repo_id=f"{HF_USER}/{TRAIN_NAME}",  # Add repo_id to satisfy validation
        push_to_hub=False,  # Disable pushing to hub
        optimizer_lr=3e-5,
        optimizer_lr_backbone=3e-5,
        # drop_n_last_frames=0,  # HACK for pick-up -> in lerobot-train change EpisodeAwareSampler end_of_episode idx
        use_tactile=False,
        tactile_input_shape=(3, 40, 40),
        tactile_features=["observation.tactile.thumb", "observation.tactile.index"],
    )

    # NOTE lerobot did some delta_timestep setup here

    # Create wandb config
    wandb_config = WandBConfig(enable=True, disable_artifact=True, project=f"{WANDB_PROJECT}", run_id=f"{TRAIN_NAME}")

    # Create training pipeline config
    config = TrainPipelineConfig(
        dataset=dataset_config,
        env=None,  # No environment for offline training
        policy=policy_config,
        output_dir=output_directory,
        job_name=f"{TRAIN_NAME}{SEED}",
        batch_size=32,  # TODO optimize this based on GPU memory
        steps=200_000,
        save_freq=20_000,
        wandb=wandb_config,
        seed=SEED,
    )

    print(f"Dataset: {config.dataset.repo_id}")
    print(f"Output dir: {config.output_dir}")

    train(config)

    print("Training completed!")


if __name__ == "__main__":
    main()
