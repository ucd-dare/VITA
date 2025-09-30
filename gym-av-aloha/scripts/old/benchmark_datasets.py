import time
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from torch.utils.data import DataLoader
from gym_av_aloha.common.replay_buffer import ReplayBuffer
from torchvision.transforms import Resize
import torch
import os
from tqdm import tqdm
from gym_av_aloha.datasets.av_aloha_dataset import AVAlohaImageDataset

repo_id = "iantc104/av_aloha_sim_peg_insertion"
meta_ds = LeRobotDatasetMetadata(repo_id)

delta_timestamps = {
    "observation.images.zed_cam_left": [t / meta_ds.fps for t in range(1 - 2, 1)],
    "observation.images.zed_cam_right": [t / meta_ds.fps for t in range(1 - 2, 1)],
    "observation.images.wrist_cam_left": [t / meta_ds.fps for t in range(1 - 2, 1)],
    "observation.images.wrist_cam_right": [t / meta_ds.fps for t in range(1 - 2, 1)],
    "observation.images.worms_eye_cam": [t / meta_ds.fps for t in range(1 - 2, 1)],
    "observation.images.overhead_cam": [t / meta_ds.fps for t in range(1 - 2, 1)],
    "observation.state": [t / meta_ds.fps for t in range(1 - 2, 1)],
    # "observation.state": [t / dataset.fps for t in range(1 - 10, 1)],
    "action": [t / meta_ds.fps for t in range(16)],
}

# Create the AVAlohaImageDataset
zarr_path = os.path.join("outputs", repo_id)
zarr_dataset = AVAlohaImageDataset(zarr_path=zarr_path, delta_timestamps=delta_timestamps)
zarr_dataloader = DataLoader(zarr_dataset, batch_size=32, shuffle=True)

lerobot_dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
lerobot_dataloader = DataLoader(lerobot_dataset, batch_size=32, shuffle=True)

print("Testing zarr dataset")
avg_time = 0
for i in range(10):
    start_time = time.time()
    zarr_batch = next(iter(zarr_dataloader))
    end_time = time.time()
    avg_time += (end_time - start_time)

print(f"Zarr dataset average time per batch: {avg_time / 64:.4f} seconds")

print("Testing lerobot dataset")
avg_time = 0
for i in range(10):
    start_time = time.time()
    lerobot_batch = next(iter(lerobot_dataloader))
    end_time = time.time()
    avg_time += (end_time - start_time)
print(f"Lerobot dataset average time per batch: {avg_time / 64:.4f} seconds")
