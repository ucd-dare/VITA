from typing import Iterator, Union

import torch
from pprint import pformat

from omegaconf import DictConfig, OmegaConf
from gym_av_aloha.datasets.av_aloha_dataset import AVAlohaDataset, AVAlohaDatasetMeta

import logging

logger = logging.getLogger(__name__)



class EpisodeAwareSampler:
    def __init__(
        self,
        episode_data_index: dict,
        episode_indices_to_use: Union[list, None] = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            episode_data_index: Dictionary with keys 'from' and 'to' containing the start and end indices of each episode.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(episode_data_index["from"], episode_data_index["to"], strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(
                    range(start_index.item() + drop_n_first_frames, end_index.item() - drop_n_last_frames)
                )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


def cycle(iterable):
    """The equivalent of itertools.cycle, but safe for Pytorch dataloaders.

    See https://github.com/pytorch/pytorch/issues/23900 for information on why itertools.cycle is not safe.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def create_dataset_stats(cfg: DictConfig):
    """Create dataset metadata and statistics."""
    dataset_meta = AVAlohaDatasetMeta(
        repo_id=cfg.task.dataset_repo_id,
        root=cfg.task.dataset_root
    )
    stats = dataset_meta.stats
    stats.update(cfg.task.override_stats)
    return dataset_meta, stats


def create_dataset(policy, cfg: DictConfig, episodes=None):
    logger.info(f"Creating dataset with repo_id={cfg.task.dataset_repo_id} and root={cfg.task.dataset_root}")

    delta_timestamps = {
        **{
            k: [i / cfg.task.fps for i in policy.get_observation_indices()]
            for k in cfg.task.image_keys
        },
        cfg.task.state_key: [i / cfg.task.fps for i in policy.get_observation_indices()],
        cfg.task.action_key: [i / cfg.task.fps for i in policy.get_action_indices()],
    }

    logger.info(f"Delta timestamps:\n{pformat(delta_timestamps, indent=4)}")

    dataset = AVAlohaDataset(
        repo_id=cfg.task.dataset_repo_id,
        root=cfg.task.dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
    )
    return dataset


def create_dataloader(dataset, cfg: DictConfig, is_training=True):
    if hasattr(cfg, "policy"):
        drop_n_last_frames = cfg.policy.get(
            "drop_n_last_frames",
            cfg.policy.pred_horizon - cfg.policy.action_horizon - cfg.policy.obs_horizon + 1
        )
    else:
        drop_n_last_frames = cfg.network.get(
            "drop_n_last_frames",
            cfg.network.pred_horizon - cfg.network.action_horizon - cfg.network.obs_horizon + 1
        )
    if drop_n_last_frames > 0:
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=drop_n_last_frames,
            shuffle=is_training,
        )
    else:
        shuffle = is_training
        sampler = None

    batch_size = cfg.train.batch_size if is_training else cfg.val.batch_size

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.train.num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader

