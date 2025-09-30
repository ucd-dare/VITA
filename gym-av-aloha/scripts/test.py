import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from gym_av_aloha.datasets.av_aloha_dataset import AVAlohaDataset
import imageio
from lerobot.common.datasets.sampler import EpisodeAwareSampler

# AVALOHADataset Test
delta_timestamps = {
    "observation.images.zed_cam_left": [0],
    "observation.state": list(range(10)),
}
dataset = AVAlohaDataset(
    repo_id="iantc104/av_aloha_sim",
    delta_timestamps=delta_timestamps,
)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

images = []
for batch in tqdm(dataloader):
    images.extend((batch["observation.images.zed_cam_left"].squeeze(1).permute(0, 2, 3, 1).numpy() * 255.0).astype("uint8"))
imageio.mimwrite(
    "av_aloha_sim.mp4",
    images,
    fps=dataset.fps,
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
    images.extend((batch["observation.images.zed_cam_left"].squeeze(1).permute(0, 2, 3, 1).numpy() * 255.0).astype("uint8"))
imageio.mimwrite(
    "av_aloha_sim_drop100.mp4",
    images,
    fps=dataset.fps,
)