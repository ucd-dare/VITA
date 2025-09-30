import torch
from pathlib import Path
import os
import numpy as np
from typing import Callable
import gym_av_aloha
from gym_av_aloha.common.replay_buffer import ReplayBuffer
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_episode_data_index,
    check_timestamps_sync,
)
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Resize
from tqdm import tqdm
from lerobot.common.datasets.compute_stats import aggregate_stats
import shutil
import json

ROOT = Path(os.path.dirname(os.path.dirname(gym_av_aloha.__file__))) / "outputs"

def make_json_serializable(obj):
    """Convert an object to a JSON-serializable format."""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def create_av_aloha_dataset_from_lerobot(
    episodes: dict[str, list[int]] | None = None,
    repo_id: str | None = None,
    root: str | Path | None = None,
    image_size: tuple[int, int] | None = None,
    remove_keys: list[str] = [],
):
    root = Path(root) if root else ROOT / repo_id
    # create lerobot datasets
    datasets = [LeRobotDataset(repo_id=repo_id, episodes=episodes) for repo_id, episodes in episodes.items()]
    # Disable any data keys that are not common across all of the datasets.
    disabled_features = set()
    intersection_features = set(datasets[0].features)
    for ds in datasets:
        intersection_features.intersection_update(ds.features)
    if len(intersection_features) == 0:
        raise RuntimeError(
            "Multiple datasets were provided but they had no keys common to all of them. "
            "The multi-dataset functionality currently only keeps common keys."
        )
    for ds in datasets:
        extra_keys = set(ds.features).difference(intersection_features)
        if len(extra_keys) > 0:
            print(
                f"keys {extra_keys} of {ds.repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
        disabled_features.update(extra_keys)
    print(
        f"Disabled features: {disabled_features}.\n"
    )
    # fps
    fps = datasets[0].meta.fps
    assert all(dataset.meta.fps == fps for dataset in datasets), "Datasets have different fps values."
    # num frames
    num_frames = sum(d.num_frames for d in datasets)
    # num episodes
    num_episodes = sum(d.num_episodes for d in datasets)
    # features
    features = {}
    for dataset in datasets:
        features.update({k: v for k, v in dataset.features.items()})
    features = {k: v for k, v in features.items() if k not in disabled_features}
    features = {k: v for k, v in features.items() if k not in remove_keys}
    # camera keys
    camera_keys = set([])
    for dataset in datasets:
        camera_keys.update(dataset.meta.camera_keys)
    camera_keys = [k for k in camera_keys if k in features]
    # video keys
    video_keys = set([])
    for dataset in datasets:
        video_keys.update(dataset.meta.video_keys)
    video_keys = [k for k in video_keys if k in features]
    # image keys
    image_keys = set([])
    for dataset in datasets:
        image_keys.update(dataset.meta.image_keys)
    image_keys = [k for k in image_keys if k in features]
    # stats
    episodes_stats = []
    for dataset in datasets:
        ep = dataset.episodes if dataset.episodes else range(dataset.num_episodes)
        for ep_idx in ep:
            episodes_stats.append({k: v for k, v in dataset.meta.episodes_stats[ep_idx].items() if k in features})
    stats = aggregate_stats(episodes_stats)
    # tasks
    tasks = []
    for ds in datasets:
        tasks.extend(ds.meta.tasks.values())
    tasks = {i: task for i, task in enumerate(tasks)}
    tasks_reversed = {v: k for k, v in tasks.items()}

    # remove old replay buffer if it exists
    if root.exists():
        print(f"Removing existing directory {root}...")
        shutil.rmtree(root)

    # create new replay buffer
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=root, mode="a")
    # metadata
    config = {
        "repo_id": dataset.repo_id,
        "stats": stats,
        "num_frames": num_frames,
        "num_episodes": num_episodes,
        "features": features,
        "camera_keys": camera_keys,
        "video_keys": video_keys,
        "image_keys": image_keys,
        "fps": fps,
        "tasks": tasks,
    }
    config_path = root / "config.json"
    with open(config_path, "w") as f: 
        json.dump(make_json_serializable(config), f, indent=4)
        
    def convert(k, v: torch.Tensor):
        dtype = features[k]['dtype']
        if dtype in ['image', 'video']:
            if image_size is not None:
                v = Resize(image_size)(v)
            # (B, C, H, W) to (B, H, W, C)
            v = v.permute(0, 2, 3, 1)
            # convert from torch float32 to numpy uint8
            v = (v * 255).to(torch.uint8).numpy()
        else:
            v = v.numpy()
        return v
        
    # iterate through dataset
    episode_idx = 0
    for dataset in datasets:
        for i in range(dataset.num_episodes):
            print(f"Converting episode {episode_idx}...")
            from_idx = dataset.episode_data_index['from'][i]
            to_idx = dataset.episode_data_index['to'][i]
            subset = Subset(dataset, range(from_idx, to_idx))
            dataloader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=8)
            data = []
            for batch in tqdm(dataloader):
                if 'task_index' in batch:
                    batch['task_index'] = torch.tensor([tasks_reversed[k] for k in batch['task']], dtype=int)
                    del batch["task"]
                batch['episode_index'] = torch.full_like(batch['episode_index'], episode_idx)
                data.append(batch)
            # since batch is a dict go through keys and cat them into a batch
            batch = {k: torch.cat([d[k] for d in data], dim=0) for k in data[0].keys()}
            assert batch['action'].shape[0] == to_idx - from_idx, f"Batch size does not match episode length. Expected {to_idx - from_idx}, got {batch['action'].shape[0]}."
            batch = {k:convert(k,v) for k,v in batch.items() if k in features}
            replay_buffer.add_episode(batch, compressors='disk')
            print(f"Episode {episode_idx} converted and added to replay buffer.")
            episode_idx += 1
    print(f"Converted dataset saved to {root}.")

def get_dataset_config(
    repo_id: str | None = None,
    root: str | Path | None = None,
) -> LeRobotDatasetMetadata:
    root = Path(root) if root else ROOT / repo_id
    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}. Please create the dataset first.")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

class AVAlohaDatasetMeta():
    def __init__(self, repo_id: str | None = None, root: str | Path | None = None):
        self.repo_id = repo_id
        self.root = Path(root) if root else ROOT / repo_id
        self.config = get_dataset_config(repo_id=self.repo_id, root=self.root)

        # convert config['tasks'] keys to int, not string
        if 'tasks' in self.config:
            self.config['tasks'] = {int(k): v for k, v in self.config['tasks'].items()}

    @property
    def stats(self) -> dict:
        return self.config['stats']
    
    @property
    def num_frames(self) -> int:
        return self.config['num_frames']
    
    @property
    def num_episodes(self) -> int:
        return self.config['num_episodes']
    
    @property
    def features(self):
        return self.config['features']
    
    @property
    def camera_keys(self):
        return self.config['camera_keys']
    
    @property
    def video_keys(self):
        return self.config['video_keys']
    
    @property
    def image_keys(self):
        return self.config['image_keys']
    
    @property
    def fps(self) -> float:
        return self.config['fps']
    
    @property
    def tasks(self):
        return self.config['tasks']

class AVAlohaDataset(torch.utils.data.Dataset):
    def __init__(self,
                 repo_id: str | None = None,
                 root: str | Path | None = None,
                 episodes: list[int] | None = None,
                 image_transforms: Callable | None = None,
                 delta_timestamps: dict[list[float]] | None = None,
                 tolerance_s: float = 1e-4,
                 ):
        super().__init__()

        self.repo_id = repo_id
        self.root = Path(root) if root else ROOT / repo_id
        self.episodes = episodes
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.episodes = episodes

        # create zarr dataset + lerobot metadata
        self.replay_buffer = ReplayBuffer.copy_from_path(self.root)
        self.meta = AVAlohaDatasetMeta(repo_id=self.repo_id, root=self.root)

        # if no episodes are specified, use all episodes in the replay buffer
        if not self.episodes: 
            self.episodes = list(range(self.meta.num_episodes))

        # calculate length of the dataset
        self.length = sum([self.replay_buffer.episode_lengths[i] for i in self.episodes])

        # add task index to delta timestamps
        if 'task_index' in self.features:
            self.delta_timestamps['task_index'] = [0]  

        # from and to indices for episodes
        self.episode_data_index = get_episode_data_index({
            i: {'length': length}
            for i, length in enumerate(self.replay_buffer.episode_lengths)
        }, self.episodes)

        # Check timestamps
        timestamps = np.array(self.replay_buffer['timestamp'])
        episode_indices = np.array(self.replay_buffer['episode_index'])
        # keep only timestamps and episode_indices for the selected episodes
        if self.episodes is not None:
            mask = np.isin(episode_indices, self.episodes)
            timestamps = timestamps[mask]
            episode_indices = episode_indices[mask]
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_replay_buffer(self, query_indices: list[str, list[int]]) -> dict:
        return {
            key: self.replay_buffer[key][q_idx]
            for key, q_idx in query_indices.items()
        }

    @property
    def stats(self):
        return self.meta.stats

    @property
    def features(self):
        return self.meta.features

    @property
    def fps(self) -> float:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        return self.meta.num_frames

    @property
    def num_episodes(self) -> int:
        return self.meta.num_episodes

    @property
    def video_keys(self):
        return self.meta.video_keys

    @property
    def image_keys(self):
        return self.meta.image_keys

    @property
    def camera_keys(self):
        return self.meta.camera_keys
    
    @property
    def tasks(self):
        return self.meta.tasks

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx = self.replay_buffer["episode_index"][idx]
        item = {"episode_index": torch.tensor(ep_idx)}

        query_indices, padding = self._get_query_indices(idx, ep_idx)
        query_result = self._query_replay_buffer(query_indices)
        item = {**item, **padding}
        for key, val in query_result.items():
            if key in self.image_keys or key in self.video_keys:
                item[key] = torch.from_numpy(val).type(torch.float32).permute(0, 3, 1, 2) / 255.0
            else:
                item[key] = torch.from_numpy(val)

        if self.image_transforms is not None:
            image_keys = self.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        if "task_index" in item:
            task_idx = item["task_index"].item()
            item["task"] = self.tasks[task_idx]

        return item
