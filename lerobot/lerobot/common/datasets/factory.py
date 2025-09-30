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
import logging
from pprint import pformat

import torch

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from gym_av_aloha.datasets import AVAlohaDataset, get_dataset_config
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, features
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the the resulting dict is empty.
    """
    fps = cfg.fps

    print(f"(resolve_delta_timestamps) fps: {fps}")

    delta_timestamps = {}
    for key in features:
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / fps for i in cfg.reward_delta_indices]
        if key == "action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and key in cfg.observation_keys and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / fps for i in cfg.observation_delta_indices]
        if key in ["left_eye", "right_eye"]:
            delta_timestamps[key] = [i / fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | AVAlohaDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        
        if cfg.dataset.use_lerobot_dataset:
            ds_meta = LeRobotDatasetMetadata(
                cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
            )
            delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta.features)
            logging.info(f"Resolved delta timestamps: {pformat(delta_timestamps, indent=2)}")
            logging.warning("Using LeRobotDataset. Please use AVAlohaImageDataset for faster training.")
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
            )
        else:
            ds_config = get_dataset_config(repo_id=cfg.dataset.repo_id)
            delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_config['features'])
            logging.info(f"Resolved delta timestamps: {pformat(delta_timestamps, indent=2)}")
            dataset = AVAlohaDataset(
                cfg.dataset.repo_id,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
            )
    elif isinstance(cfg.dataset.repo_id, list) and len(cfg.dataset.repo_id) > 0:
        if cfg.dataset.use_lerobot_dataset:
            raise NotImplementedError(
                "MultiLeRobotDataset is currently deactivated. Please use AVAlohaImageDataset for multiple datasets."
            )
            dataset = MultiLeRobotDataset(
                cfg.dataset.repo_id,
                # TODO(aliberts): add proper support for multi dataset
                # delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                video_backend=cfg.dataset.video_backend,
            )
            logging.info(
                "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
                f"{pformat(dataset.repo_id_to_index, indent=2)}"
            )
    else:
        raise ValueError(f"Unsupported dataset repo_id type: {cfg.dataset.repo_id}")

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
