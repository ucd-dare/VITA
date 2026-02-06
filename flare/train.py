import os
import torch
import hydra
import logging
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

from flare.factory import get_policy_class, get_runner_class
from flare.utils.checkpoints import get_latest_checkpoint

from flare.utils.eval import create_eval_env
from flare.utils.logger import setup_logging
from flare.utils.dataset_utils import create_dataloader, create_dataset, create_dataset_stats

logger = logging.getLogger(__name__)


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
        eval_env = create_eval_env(
            eval_n_envs=cfg.val.eval_n_envs,
            env_package=cfg.task.env_package,
            env_name=cfg.task.env_name,
            env_kwargs=cfg.task.env_kwargs
        )

    runner_cls = get_runner_class(cfg.policy.runner)
    runner = runner_cls(
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
            start_step = runner.load_checkpoint(checkpoint_path)
            logger.info(f"Resumed from checkpoint at step {start_step}")
        else:
            logger.warning(f"No checkpoint found, starting from scratch")

    # Training loop
    runner.train(
        num_steps=cfg.train.steps,
        start_step=start_step,
    )

    # Cleanup
    if eval_env:
        eval_env.close()

    logger.info("Training completed!")


if __name__ == "__main__":
    train()
