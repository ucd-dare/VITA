import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from gym_av_aloha.datasets.av_aloha_dataset import AVAlohaDataset
from gym_av_aloha.datasets.multi_av_aloha_dataset import MultiAVAlohaDataset
import imageio
from lerobot.common.datasets.sampler import EpisodeAwareSampler

# PUSHT AVALOHADataset

delta_timestamps = {
    "observation.image": [0],
}
dataset = AVAlohaDataset(
    repo_id="lerobot/pusht", 
    delta_timestamps=delta_timestamps,
    episodes=[0, 10, 20],
)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

avg_time = 0
for i in range(10):
    start_time = time.time()
    batch = next(iter(dataloader))
    end_time = time.time()
    avg_time += (end_time - start_time)
print(f"AV ALOHA Dataset average time per batch: {avg_time / 10:.4f} seconds")

images = []
for batch in tqdm(dataloader):
    images.extend((batch["observation.image"].squeeze(1).permute(0, 2, 3, 1).numpy() * 255.0).astype("uint8"))
imageio.mimwrite(
    "pusht.mp4",
    images,
    fps=dataset.meta.fps,
)

# AVAlohaDataset Sampler Test
sampler = EpisodeAwareSampler(
    dataset.episode_data_index,
    drop_n_last_frames=100,
    shuffle=False,
)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    sampler=sampler,
)
images = []
for batch in tqdm(dataloader):
    images.extend((batch["observation.image"].squeeze(1).permute(0, 2, 3, 1).numpy() * 255.0).astype("uint8"))
imageio.mimwrite(
    "pusht_drop30.mp4",
    images,
    fps=dataset.meta.fps,
)

# MultiAVAlohaDataset
delta_timestamps = {
    "observation.images.zed_cam_left": [0],
    "observation.images.zed_cam_right": [t / 25 for t in range(1 - 2, 1)],
    "observation.state": [t / 25 for t in range(1 - 2, 1)],
    "action": [t / 25 for t in range(16)],
}
dataset = MultiAVAlohaDataset(
    repo_ids=["iantc104/av_aloha_sim_peg_insertion_240x320", "iantc104/av_aloha_sim_thread_needle_240x320"],
    delta_timestamps=delta_timestamps,
    episodes={
        "iantc104/av_aloha_sim_peg_insertion_240x320": [49],
        "iantc104/av_aloha_sim_thread_needle_240x320": [56, 199],
    }
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
images = []
for batch in tqdm(dataloader):
    images.extend((batch["observation.images.zed_cam_left"].squeeze(1).permute(0, 2, 3, 1).numpy() * 255.0).astype("uint8"))
# save mp4
imageio.mimwrite(
    "avaloha.mp4",
    images,
    fps=dataset.fps,
)

# Task test

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
batch = next(iter(dataloader))
print(batch['task'])


# Test MultiAVAlohaDataset with sampler

sampler = EpisodeAwareSampler(
    dataset.episode_data_index,
    drop_n_last_frames=100,
    shuffle=False,
)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    sampler=sampler,
)
images = []
for batch in tqdm(dataloader):
    images.extend((batch["observation.images.zed_cam_left"].squeeze(1).permute(0, 2, 3, 1).numpy() * 255.0).astype("uint8"))
# save mp4
imageio.mimwrite(
    "avaloha_drop100.mp4",
    images,
    fps=dataset.fps,
)