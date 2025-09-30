import logging
from pathlib import Path
from typing import Callable
import datasets
import torch
import torch.utils
import numpy as np
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.video_utils import VideoFrame
from gym_av_aloha.datasets.av_aloha_dataset import AVAlohaDataset, ROOT

# BROKEN AND SUCKS

class MultiAVAlohaDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else ROOT
        self.tolerances_s = tolerances_s if tolerances_s else {repo_id: 1e-4 for repo_id in repo_ids}
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = [
            AVAlohaDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
            )
            for repo_id in repo_ids
        ]

        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            if len(extra_keys) > 0:
                logging.warning(
                    f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                    "other datasets."
                )
            self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
        # with multiple robots of different ranges. Instead we should have one normalization
        # per robot.
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info.get("video", False)

    @property
    def hf_features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features})
        return features
    
    @property
    def features(self) -> dict:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.features.items() if k not in self.disabled_features})
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = set([])
        for dataset in self._datasets:
            keys.update(dataset.meta.camera_keys)
        return list(keys)

    @property
    def video_keys(self) -> list[str]:
        """Keys to access video frames from cameras."""
        keys = set([])
        for dataset in self._datasets:
            keys.update(dataset.meta.video_keys)
        return list(keys)

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4
    
    @property
    def episode_data_index(self) -> dict:
        """A dictionary mapping episode indices to the start and end frame indices of each episode."""
        from_idx = []
        to_idx = []
        start_idx = 0
        for dataset in self._datasets:
            from_idx.extend(dataset.episode_data_index["from"] + start_idx)
            to_idx.extend(dataset.episode_data_index["to"] + start_idx)
            start_idx += dataset.num_frames

        episode_data_index = {
            "from": torch.tensor(from_idx),
            "to": torch.tensor(to_idx),
        }
        valid_indices = np.concatenate([
            np.arange(episode_data_index["from"][ep], episode_data_index["to"][ep])
            for ep in range(len(episode_data_index["from"]))
        ])
        assert len(valid_indices) == self.num_frames, 'Length of valid indices does not match length of dataset.'
        assert len(set(valid_indices)) == len(valid_indices), 'Valid indices contain duplicates.'
        return episode_data_index


    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
