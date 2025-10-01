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

import abc
from dataclasses import dataclass, field

import draccus

from lerobot.common.constants import ACTION, OBS_ENV, OBS_IMAGE, OBS_IMAGES, OBS_ROBOT
from lerobot.configs.types import FeatureType, PolicyFeature


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: float = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractproperty
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()


AV_ONLY = True

AV_CAM_FACTORY = field(default_factory=lambda: {
    "zed_cam_left": [480, 640],
    "zed_cam_right": [480, 640],
})

FULL_CAM_FACTORY = field(default_factory=lambda: {
    "zed_cam_left": [480, 640],
    "zed_cam_right": [480, 640],
    "wrist_cam_left": [480, 640],
    "wrist_cam_right": [480, 640],
    "overhead_cam": [480, 640],
    "worms_eye_cam": [480, 640],
})

AV_CAM_FEATURES_MAP_FACTORY = field(default_factory=lambda: {
    "action": ACTION,
    "agent_pos": OBS_ROBOT,
    'zed_cam_left': f"{OBS_IMAGE}.zed_cam_left",
    'pixels/zed_cam_left': f"{OBS_IMAGES}.zed_cam_left",
    'zed_cam_right': f"{OBS_IMAGE}.zed_cam_right",
    'pixels/zed_cam_right': f"{OBS_IMAGES}.zed_cam_right",
    "environment_state": OBS_ENV,
})

FULL_CAM_FEATURES_MAP_FACTORY = field(default_factory=lambda: {
    "action": ACTION,
    "agent_pos": OBS_ROBOT,
    "overhead_cam": f"{OBS_IMAGE}.overhead_cam",
    "pixels/overhead_cam": f"{OBS_IMAGES}.overhead_cam",
    'worms_eye_cam': f"{OBS_IMAGE}.worms_eye_cam",
    'pixels/worms_eye_cam': f"{OBS_IMAGES}.worms_eye_cam",
    'zed_cam_left': f"{OBS_IMAGE}.zed_cam_left",
    'pixels/zed_cam_left': f"{OBS_IMAGES}.zed_cam_left",
    'zed_cam_right': f"{OBS_IMAGE}.zed_cam_right",
    'pixels/zed_cam_right': f"{OBS_IMAGES}.zed_cam_right",
    'wrist_cam_left': f"{OBS_IMAGE}.wrist_cam_left",
    'pixels/wrist_cam_left': f"{OBS_IMAGES}.wrist_cam_left",
    'wrist_cam_right': f"{OBS_IMAGE}.wrist_cam_right",
    'pixels/wrist_cam_right': f"{OBS_IMAGES}.wrist_cam_right",
    "environment_state": OBS_ENV,
})


@EnvConfig.register_subclass("av_aloha")
@dataclass
class AVAlohaEnv(EnvConfig):
    task: str = "peg-insertion-v1"

    fps: float = 25
    enable_av: bool = True
    episode_length_s: float = 10
    episode_length: int = -1
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    render_camera: str = "zed_cam_right"
    cameras: dict[str, list[int]] = AV_CAM_FACTORY if AV_ONLY else FULL_CAM_FACTORY
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(21,)),
        }
    )
    features_map: dict[str, str] = AV_CAM_FEATURES_MAP_FACTORY if AV_ONLY else FULL_CAM_FEATURES_MAP_FACTORY

    def __post_init__(self):
        print("Initializing AVAlohaEnv with cameras:", self.cameras)

        self.episode_length = int(self.episode_length_s * self.fps)
        print(f"(AVAlohaEnv) episode_length: {self.episode_length}")

        if not self.enable_av:
            # Reduce action dimension by 7 to exclude middle arm
            self.features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(14,))
            if self.render_camera in ["zed_cam_left", "zed_cam_right"]:
                self.render_camera = "worms_eye_cam"

        if self.obs_type == "pixels":
            for cam in self.cameras:
                self.features[f"{cam}"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
        elif self.obs_type == "pixels_agent_pos":
            if self.enable_av:
                self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(21,))
            else:
                self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))

            for cam in self.cameras:
                self.features[f"pixels/{cam}"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "fps": self.fps,
            "cameras": self.cameras,
            "render_camera": self.render_camera,
            "enable_av": self.enable_av,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    task: str = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "top": f"{OBS_IMAGE}.top",
            "pixels/top": f"{OBS_IMAGES}.top",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("pusht")
@dataclass
class PushtEnv(EnvConfig):
    task: str = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "environment_state": OBS_ENV,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["pixels"] = PolicyFeature(type=FeatureType.VISUAL, shape=(384, 384, 3))
        elif self.obs_type == "environment_state_agent_pos":
            self.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(16,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("xarm")
@dataclass
class XarmEnv(EnvConfig):
    task: str = "XarmLift-v0"
    fps: int = 15
    episode_length: int = 200
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(84, 84, 3)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }
