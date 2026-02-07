import torch
import logging
from pathlib import Path
from collections import defaultdict

from flare.factory import registry
from flare.utils.logger import FlareLogger
from flare.utils.eval import eval_policy
from flare.utils.checkpoints import load_model_weights

logger = logging.getLogger(__name__)


@registry.register_runner("policy_evaluator")
class PolicyEvaluator:
    def __init__(
        self,
        config,
        network,
        device,
        val_dataloader=None,
        eval_env=None,
    ):
        self.config = config
        self.network = network.to(device)
        self.device = device
        self.val_dataloader = val_dataloader
        self.eval_env = eval_env
        self.ema = network.get_ema()

        self.flare_logger = FlareLogger(config)

    def validate_offline(self):
        self.network.eval()
        val_metrics = defaultdict(float)
        num_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
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
                        )

        val_metrics = {k: v / num_samples for k, v in val_metrics.items()}
        return val_metrics

    def validate_online(self):
        model = self.network

        if self.config.use_ema and self.ema is not None:
            logger.info("Using EMA weights for evaluation")
            self.ema.store(model.parameters())
            self.ema.copy_to(model.parameters())

        model.eval()
        with torch.no_grad():
            eval_info = eval_policy(
                self.eval_env,
                model,
                self.config.val.eval_n_episodes,
                videos_dir=Path(self.config.eval_dir) / f"videos",
                max_episodes_rendered=self.config.val.num_viz_videos,
                start_seed=self.config.seed,
            )

        if self.config.use_ema and self.ema is not None:
            self.ema.restore(model.parameters())

        return eval_info

    def eval(self):
        offline_metrics = self.validate_offline()
        online_metrics = self.validate_online()
        logger.info(f"Online metrics: {online_metrics}")
        logger.info(f"Offline metrics: {offline_metrics}")
        aggregated = online_metrics.get("aggregated", {})
        per_episode = online_metrics.get("per_episode", [])
        num_episodes = len(per_episode)

        summary = {
            "episodes": num_episodes,
            "pc_success": aggregated.get("pc_success"),
            "avg_max_reward": aggregated.get("avg_max_reward"),
            "loss": offline_metrics.get("loss"),
            "action_mse": offline_metrics.get("action_mse"),
            "eval_s": aggregated.get("eval_s"),
            "eval_ep_s": aggregated.get("eval_ep_s"),
        }
        logger.info(f"Eval summary: {summary}")

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)

        training_state_file = checkpoint_dir / "training_state.pt"

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        if not training_state_file.exists():
            raise FileNotFoundError(f"Training state file not found: {training_state_file}")

        load_model_weights(self.network, checkpoint_dir, self.device)
        self.network.to(self.device)

        training_state = torch.load(
            training_state_file,
            map_location=self.device,
            weights_only=False
        )

        if self.ema and training_state.get("ema"):
            self.ema.load_state_dict(training_state["ema"])

        step = training_state.get("step")

        logger.info(f"Loaded checkpoint from {checkpoint_dir} at step {step}")
        return step
