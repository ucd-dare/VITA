import torch
import hydra
import logging
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

from flare.factory import get_policy_class
from flare.runners.policy_evaluator import PolicyEvaluator
from flare.utils.dataset_utils import create_dataloader, create_dataset, create_dataset_stats
from flare.utils.checkpoints import get_best_checkpoint, get_latest_checkpoint
from flare.utils.eval import create_eval_env
from flare.utils.logger import setup_logging


logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="default_policy", version_base="1.3")
def evaluate(cfg: DictConfig):

    # Setup
    setup_logging(save_dir=cfg.log_dir, log_file="eval.log", debug=cfg.debug)
    logger.info("Starting evaluation...")
    logger.info(f"Using policy: {cfg.policy.name}")

    if cfg.checkpoint_path:
        logger.info(f"Searching for the specified checkpoint path: {cfg.checkpoint_path}")
        checkpoint_path = Path(cfg.checkpoint_path)
    else:
        logger.info(f"Searching for best checkpoint in {cfg.checkpoint_dir}")
        checkpoint_path = get_best_checkpoint(cfg.checkpoint_dir)
        if checkpoint_path is None:
            logger.info(f"Searching for latest checkpoint in {cfg.checkpoint_dir}")
            checkpoint_path = get_latest_checkpoint(cfg.checkpoint_dir)
    if checkpoint_path is None:
        logger.warning(f"No checkpoint found")
        return

    # Set seeds
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Create directories and save config
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d%H%M")
    OmegaConf.save(config=cfg, f=cfg.log_dir / Path(f"eval_config_{time_str}.yaml"))

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create dataset stats
    dataset_meta, stats = create_dataset_stats(cfg)

    # Create policy
    policy_cls = get_policy_class(cfg.policy.name)
    policy = policy_cls(cfg, stats)
    num_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Number of parameters: {num_params/1e6:.2f}M")

    # Create dataset and dataloader
    total_episodes = dataset_meta.num_episodes
    if cfg.val.num_episodes is not None and cfg.val.num_episodes > 0:
        val_dataset = create_dataset(policy, cfg, list(range(total_episodes - cfg.val.num_episodes, total_episodes)))
        val_dataloader = create_dataloader(val_dataset, cfg, is_training=False)

        logger.info(f"Val: {val_dataset.num_frames} frames | {val_dataset.num_episodes} episodes | {len(val_dataloader)} batches")
    else:
        val_dataloader = None

    eval_env = create_eval_env(
        eval_n_envs=cfg.val.eval_n_envs,
        env_package=cfg.task.env_package,
        env_name=cfg.task.env_name,
        env_kwargs=cfg.task.env_kwargs
    )

    evaluator = PolicyEvaluator(
        cfg,
        policy,
        device=cfg.device,
        val_dataloader=val_dataloader,
        eval_env=eval_env
    )

    start_step = evaluator.load_checkpoint(checkpoint_path)
    logger.info(f"Evaluating checkpoint at path {checkpoint_path} and step {start_step}")

    evaluator.eval()

    # Cleanup
    if eval_env:
        eval_env.close()

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    evaluate()
