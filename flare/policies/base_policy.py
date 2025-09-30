import time
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from diffusers.training_utils import EMAModel
from huggingface_hub import PyTorchModelHubMixin

from flare.utils.normalize import Normalize, Unnormalize

logger = logging.getLogger(__name__)


class BasePolicy(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config, stats):
        super().__init__()

        self.config = config
        self.pred_horizon = config.policy.pred_horizon
        self.action_horizon = config.policy.action_horizon
        self.obs_horizon = config.policy.obs_horizon

        self.normalize_inputs = Normalize(config.task.image_keys+[config.task.state_key], stats)
        self.normalize_targets = Normalize([config.task.action_key], stats)
        self.unnormalize_outputs = Unnormalize([config.task.action_key], stats)
        self._action_queue = None

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def generate_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        loss, metrics = self.compute_loss(batch)
        return loss, metrics

    def get_ema(self):
        return EMAModel(
            parameters=self.parameters(),
            power=self.config.ema_power,
        ) if self.config.use_ema else None

    def get_action_indices(self):
        return list(range(self.pred_horizon))

    def get_observation_indices(self):
        return list(range(self.obs_horizon))

    def reset(self):
        self._action_queue = deque([], maxlen=self.action_horizon)

    def validate(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        self.eval()
        batch = self.normalize_inputs(batch)
        batch_size = batch[self.config.task.action_key].shape[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            pred_norm = self.generate_actions(batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_time_ms = (time.perf_counter() - t0) * 1000 / batch_size # milliseconds

        pred = self.unnormalize_outputs({"action": pred_norm})["action"]
        gt = batch[self.config.task.action_key]

        loss, metrics = self.compute_loss(batch)

        metrics["action_mse"] = F.mse_loss(pred, gt).item()
        metrics["loss"] = loss.item()

        total_actions = batch_size * self.action_horizon
        metrics["time_per_chunk_ms"] = gen_time_ms
        metrics["time_per_action_ms"] = gen_time_ms / total_actions
        metrics["chunks_per_sec"] = 1000 / gen_time_ms

        return metrics
