import torch
import logging
from pathlib import Path
from collections import defaultdict

from flare.factory import registry
from flare.utils.logger import FlareLogger
from flare.utils.dataset_utils import cycle
from flare.utils.logging_utils import AverageMeter, MetricsTracker
from flare.eval import eval_policy

logger = logging.getLogger(__name__)


@registry.register_trainer("policy")
class PolicyTrainer:
    def __init__(
        self,
        config,
        network,
        device,
        train_dataloader=None,
        val_dataloader=None,
        eval_env=None,
    ):
        self.config = config
        self.network = network.to(device)
        self.device = device
        self.step_counter = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.eval_env = eval_env

        num_total_steps = self.config.train.steps

        self.optimizer = network.get_optimizer()
        self.lr_scheduler = network.get_scheduler(self.optimizer, num_total_steps)
        self.ema = network.get_ema()

        self.flare_logger = FlareLogger(config)

        self.train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grad_norm", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("update_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }

        self.best_metrics = {
            "pc_success": -float("inf"),
            "avg_max_reward": -float("inf"),
        }

        dataset = train_dataloader.dataset

        self.train_tracker = MetricsTracker(
            self.config.train.batch_size,
            dataset.num_frames,
            dataset.num_episodes,
            self.train_metrics,
            initial_step=self.step_counter
        )

    def train_step(self, batch):
        """Single training step."""
        import time
        #         # --- DEBUG: print shapes once ---
        # if not hasattr(self, "_printed_batch_debug"):
        #     try:
        #         st = batch["observation.state"]
        #         actual = int(st.shape[-1])
        #         expected = int(self.config.task.state_dim)
        #         # Print state + first image key
        #         img_key = self.config.task.image_keys[0]
        #         img = batch.get(img_key, None)

        #         logging.info(
        #             f"[DBG] observation.state shape={tuple(st.shape)} "
        #             f"(last_dim={actual}, expected={expected})"
        #         )
        #         if isinstance(img, torch.Tensor):
        #             logging.info(f"[DBG] {img_key} shape={tuple(img.shape)}")
        #         else:
        #             logging.info(f"[DBG] {img_key} not in batch")

        #         # Optional hard guard: fail early on mismatch
        #         if actual != expected:
        #             raise ValueError(
        #                 f"Config task.state_dim={expected} but batch last_dim={actual}. "
        #                 f"Either set state_dim to {actual} or slice the state vector before the model."
        #             )
        #     finally:
        #         self._printed_batch_debug = True
        # # --- /DEBUG ---


        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device, non_blocking=True)

        # Forward pass
        start_time = time.perf_counter()
        self.network.train()
        self.optimizer.zero_grad()

        loss, output_dict = self.network.forward(batch)

        # Backward pass
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.config.train.grad_clip_norm,
            error_if_nonfinite=False,
        )

        # Optimizer step
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.ema is not None:
            self.ema.step(self.network.parameters())

        # Update metrics
        self.train_tracker.loss = loss.item()
        self.train_tracker.grad_norm = grad_norm.item()
        self.train_tracker.lr = self.optimizer.param_groups[0]["lr"]
        self.train_tracker.update_s = time.perf_counter() - start_time

        return output_dict

    def validate_offline(self):
        val_dataloader = self.train_dataloader
        self.network.eval()
        val_metrics = defaultdict(float)
        num_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)

                output_dict = self.network.validate(batch)

                batch_size = next(iter(batch.values())).shape[0]
                for k, v in output_dict.items():
                    val_metrics[k] += v * batch_size
                num_samples += batch_size

                if batch_idx == 0 and hasattr(self.network, 'visualize'):
                    viz_results = self.network.visualize(
                        batch,
                        num_samples=self.config.val.num_viz_samples
                    )

                    if viz_results:
                        self.flare_logger.log_figures(
                            viz_results,
                            prefix='plot',
                            step=self.step_counter
                        )

        val_metrics = {k: v / num_samples for k, v in val_metrics.items()}
        return val_metrics

    def validate_online(self):
        model = self.network

        if self.config.use_ema and self.ema is not None:
            self.ema.store(model.parameters())
            self.ema.copy_to(model.parameters())

        model.eval()
        with torch.no_grad():
            eval_info = eval_policy(
                self.eval_env,
                model,
                self.config.val.eval_n_episodes,
                videos_dir=Path(self.config.val_dir) / f"videos_step_{self.step_counter:010d}",
                max_episodes_rendered=self.config.val.num_viz_videos,
                start_seed=self.config.seed,
            )

        if self.config.use_ema and self.ema is not None:
            self.ema.restore(model.parameters())

        return eval_info

    def train(self, num_steps=None, start_step=0):
        import time

        if num_steps is None:
            num_steps = self.config.train.steps

        self.step_counter = start_step
        dl_iter = cycle(self.train_dataloader)

        logger.info(f"Starting training from step {start_step} to {num_steps}")

        for step in range(start_step, num_steps):
            # Load data
            start_time = time.perf_counter()
            batch = next(dl_iter)
            self.train_tracker.dataloading_s = time.perf_counter() - start_time

            # Train step
            output_dict = self.train_step(batch)

            # Update step counter
            self.step_counter += 1
            self.train_tracker.step()

            # Check intervals
            is_log_step = (
                self.config.train.log_freq > 0 and
                self.step_counter % self.config.train.log_freq == 0
            )
            is_save_step = (
                self.config.train.save_freq > 0 and
                self.step_counter % self.config.train.save_freq == 0
            )
            is_val_offline_step = (
                self.val_dataloader is not None and
                self.config.val.val_offline_freq > 0 and
                self.step_counter % self.config.val.val_offline_freq == 0
            )
            is_val_online_step = (
                self.eval_env and
                self.config.val.val_online_freq > 0 and
                self.step_counter % self.config.val.val_online_freq == 0
            )

            # Log metrics
            if is_log_step:
                logger.info(self.train_tracker)
                wandb_log_dict = self.train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                self.flare_logger.log_metrics(wandb_log_dict, prefix='train', step=self.step_counter)
                self.train_tracker.reset_averages()

            if is_val_offline_step:
                logger.info(f"Offline validation at step {self.step_counter}")
                val_metrics = self.validate_offline()

                if val_metrics:
                    logger.info(f"Validation metrics: {val_metrics}")
                    self.flare_logger.log_metrics(
                        val_metrics,
                        prefix='val_offline',
                        step=self.step_counter
                    )

            # Online validation
            if is_val_online_step:
                logger.info(f"Online validation at step {self.step_counter}")
                eval_info = self.validate_online()

                # Log aggregated metrics
                logger.info(f"Eval metrics: {eval_info['aggregated']}")
                self.flare_logger.log_metrics(
                    eval_info['aggregated'],
                    prefix='val_online',
                    step=self.step_counter
                )

                # Log video
                if eval_info['video_paths']:
                    self.flare_logger.log_video_files(
                        {"eval_video": eval_info['video_paths'][0]},
                        prefix='val_online',
                        fps=self.config.task.fps,
                        step=self.step_counter
                    )

                # Save best checkpoints
                if self.config.train.save_best:
                    if eval_info['aggregated']['pc_success'] > self.best_metrics['pc_success']:
                        self.best_metrics['pc_success'] = eval_info['aggregated']['pc_success']
                        self.save_checkpoint(Path(self.config.checkpoint_dir) / "best_pc_success")

                    if eval_info['aggregated']['avg_max_reward'] > self.best_metrics['avg_max_reward']:
                        self.best_metrics['avg_max_reward'] = eval_info['aggregated']['avg_max_reward']
                        self.save_checkpoint(Path(self.config.checkpoint_dir) / "best_avg_max_reward")

            if is_save_step:
                self.save_checkpoint(Path(self.config.checkpoint_dir) / f"step_{self.step_counter:010d}")

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.network.save_pretrained(checkpoint_dir)

        # Save training state
        training_state = {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "ema": self.ema.state_dict() if self.ema else None,
            "step": self.step_counter,
        }
        torch.save(training_state, checkpoint_dir / "training_state.pt")

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)

        training_state_file = checkpoint_dir / "training_state.pt"

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        if not training_state_file.exists():
            raise FileNotFoundError(f"Training state file not found: {training_state_file}")

        # Load network
        self.network = self.network.from_pretrained(checkpoint_dir)
        self.network.to(self.device)

        # Load training state
        training_state = torch.load(training_state_file, map_location=self.device)
        self.optimizer.load_state_dict(training_state["optimizer"])

        if self.lr_scheduler and training_state.get("lr_scheduler"):
            self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])

        if self.ema and training_state.get("ema"):
            self.ema.load_state_dict(training_state["ema"])

        step = training_state.get("step", 0)
        self.step_counter = step

        logger.info(f"Loaded checkpoint from {checkpoint_dir} at step {step}")
        return step
