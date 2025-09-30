import os
import torch
import hydra
import logging
import random
import importlib
import numpy as np
from pprint import pformat
from pathlib import Path
from datetime import datetime
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf

from flare.factory import get_policy_class, get_trainer_class
from flare.utils.logger import setup_logging
from flare.utils.checkpoints import get_latest_checkpoint

from gym_av_aloha.datasets.av_aloha_dataset import AVAlohaDataset, AVAlohaDatasetMeta
from flare.utils.dataset_utils import EpisodeAwareSampler

logger = logging.getLogger(__name__)


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
    logging.info(f"Creating dataset with repo_id={cfg.task.dataset_repo_id} and root={cfg.task.dataset_root}")

    delta_timestamps = {
        **{
            k: [i / cfg.task.fps for i in policy.get_observation_indices()]
            for k in cfg.task.image_keys
        },
        cfg.task.state_key: [i / cfg.task.fps for i in policy.get_observation_indices()],
        cfg.task.action_key: [i / cfg.task.fps for i in policy.get_action_indices()],
    }

    logging.info(f"Delta timestamps:\n{pformat(delta_timestamps, indent=4)}")

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


def create_eval_env(cfg: DictConfig):
    """Create evaluation environment."""
    logging.info(f"""
        Creating {cfg.val.eval_n_envs} environments
        id={cfg.task.env_package}/{cfg.task.env_name}
        kwargs={pformat(cfg.task.env_kwargs, indent=4)}
    """)

    importlib.import_module(cfg.task.env_package)
    eval_env = gym.vector.SyncVectorEnv([
        lambda: gym.make(
            f"{cfg.task.env_package}/{cfg.task.env_name}",
            **cfg.task.env_kwargs
        )
        for _ in range(cfg.val.eval_n_envs)
    ])

    return eval_env


@hydra.main(config_path="configs", config_name="default_policy", version_base="1.3")
def train(cfg: DictConfig):

    # Setup
    setup_logging(save_dir=cfg.log_dir, debug=cfg.debug)
    logger.info("Starting training...")
    logger.info(f"Using policy: {cfg.policy.name}")

    # Set seeds
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Create directories and save config
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d%H%M")
    OmegaConf.save(config=cfg, f=cfg.log_dir / Path(f"train_config_{time_str}.yaml"))

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create dataset stats
    dataset_meta, stats = create_dataset_stats(cfg)

    # Create policy
    policy_cls = get_policy_class(cfg.policy.name)
    policy = policy_cls(cfg, stats)
    num_params = sum(p.numel() for p in policy.parameters())
    num_trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params/1e6:.2f}M | Trainable params: {num_trainable_params/1e6:.2f}M")

    # Create dataset and dataloader
    total_episodes = dataset_meta.num_episodes
    if cfg.val.num_episodes != None and cfg.val.num_episodes > 0:
        train_dataset = create_dataset(policy, cfg, list(range(total_episodes - cfg.val.num_episodes)))
        train_dataloader = create_dataloader(train_dataset, cfg, is_training=True)
        val_dataset = create_dataset(policy, cfg, list(range(total_episodes - cfg.val.num_episodes, total_episodes)))
        val_dataloader = create_dataloader(val_dataset, cfg, is_training=False)

        logger.info(f"Train: {train_dataset.num_frames} frames | {train_dataset.num_episodes} episodes | {len(train_dataloader)} batches")
        logger.info(f"Val: {val_dataset.num_frames} frames | {val_dataset.num_episodes} episodes | {len(val_dataloader)} batches")
    else:
        train_dataset = create_dataset(policy, cfg)
        train_dataloader = create_dataloader(train_dataset, cfg, is_training=True)
        val_dataloader = None
        logger.info(f"Train: {train_dataset.num_frames} frames | {train_dataset.num_episodes} episodes | {len(train_dataloader)} batches")

    # Create evaluation environment
    eval_env = None
    if cfg.val.val_online_freq > 0:
        eval_env = create_eval_env(cfg)

    trainer_cls = get_trainer_class(cfg.policy.trainer)
    trainer = trainer_cls(
        cfg,
        policy,
        device=cfg.device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        eval_env=eval_env
    )

    # Resume from checkpoint if specified
    start_step = 0
    if cfg.resume:
        if cfg.checkpoint_path:
            checkpoint_path = Path(cfg.checkpoint_path)
        else:
            checkpoint_path = get_latest_checkpoint(cfg.checkpoint_dir)

        if checkpoint_path:
            start_step = trainer.load_checkpoint(checkpoint_path)
            logger.info(f"Resumed from checkpoint at step {start_step}")
        else:
            logger.warning(f"No checkpoint found, starting from scratch")

    # Training loop
    trainer.train(
        num_steps=cfg.train.steps,
        start_step=start_step,
    )

    # Cleanup
    if eval_env:
        eval_env.close()

    logger.info("Training completed!")


if __name__ == "__main__":
    train()
