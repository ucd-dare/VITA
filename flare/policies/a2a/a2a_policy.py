"""
A2A: Action-to-Action Flow Matching, Jindou Jia et al. (2026)

GitHub:  https://github.com/JIAjindou/A2A_Flow_Matching
Website: https://lorenzo-0-0.github.io/A2A_Flow_Matching/

A2A is a highly generalizable policy that flows from proprioceptive features
to action latents conditioned on visual inputs.
"""

import torch
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers.optimization import get_scheduler

from flare.factory import registry
from flare.flow.flow_matchers import get_flow_matcher
from flare.networks.vita.action_ae import get_autoencoder
from flare.networks.vita.action_vae import get_variational_autoencoder
from flare.policies import BasePolicy
from flare.policies.vita.simple_flow_net import SimpleCondFlowNet
from flare.visualizer.visualizer import plot_ode_steps, plot_trajectory
from flare.policies.observers.resnet_observer import ResNetObserver

logger = logging.getLogger(__name__)


@registry.register_policy("a2a")
class A2APolicy(BasePolicy):
    def __init__(self, config, stats):
        super().__init__(config, stats)

        self.num_sampling_steps = config.policy.flow_matcher.num_sampling_steps
        self.FM = get_flow_matcher(**config.policy.flow_matcher)

        self.observer = ResNetObserver(
            state_key=config.task.state_key,
            image_keys=config.task.image_keys,
            resize_shape=config.resize_shape,
            crop_shape=config.crop_shape,
            state_dim=config.task.state_dim,
            tokenize=False,
        )
        self.obs_dim = len(config.task.image_keys) * 512 + config.task.state_dim
        self.obs_encoder = torch.nn.Linear(self.obs_dim, config.policy.a2a.latent_dim)

        action_ae_net_config = config.policy.action_ae.net
        self.use_action_vae = config.policy.action_ae.use_variational
        self.latent_dim = action_ae_net_config.latent_dim
        self.freeze_action_encoder = config.policy.action_ae.freeze_encoder
        self.freeze_action_decoder = config.policy.action_ae.freeze_decoder
        self.flow_action_recon_weight = config.policy.action_ae.flow_recon_weight
        self.enc_action_recon_weight = config.policy.action_ae.enc_recon_weight

        self.action_kl_weight = config.policy.action_ae.kl_weight
        self.recon_loss_fn = F.l1_loss

        self.decode_flow_latents = config.policy.a2a.decode_flow_latents
        self.consistency_weight = config.policy.a2a.consistency_weight
        self.enc_contrastive_weight = config.policy.a2a.enc_contrastive_weight
        self.flow_contrastive_weight = config.policy.a2a.flow_contrastive_weight

        logger.info("Using proprioceptive state features as flow source.")
        self.state_encoder = torch.nn.Linear(self.obs_horizon * config.task.state_dim, self.latent_dim)

        self._init_action_vae(action_ae_net_config)

        logger.info("Using MLP for flow velocity prediction.")
        self.flow_net = SimpleCondFlowNet(
            input_dim=self.latent_dim,
            hidden_dim=config.policy.flow_net.hidden_dim,
            output_dim=self.latent_dim,
            num_layers=config.policy.flow_net.num_layers,
            mlp_ratio=config.policy.flow_net.mlp_ratio,
            dropout=config.policy.flow_net.dropout,
            condition_dim=self.latent_dim,
        )

        self.reset()

    def _init_action_vae(self, action_ae_net_config):
        encoder_type = action_ae_net_config.encoder_type
        decoder_type = action_ae_net_config.decoder_type
        logger.info(f"Using {encoder_type} encoder and {decoder_type} decoder for action autoencoder.")

        if self.use_action_vae:
            self.action_encoder, self.action_decoder = get_variational_autoencoder(self.config, self.config.policy)
        else:
            self.action_encoder, self.action_decoder = get_autoencoder(self.config, self.config.policy)

    def _encode_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        states = batch[self.config.task.state_key][:, : self.obs_horizon]
        return self.state_encoder(states.flatten(start_dim=1))

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state_latents = self._encode_state(batch)
        obs_features = self.observer(batch)
        visual_latents = self.obs_encoder(obs_features)

        batch_size = state_latents.shape[0]
        gt_actions = batch[self.config.task.action_key]

        with torch.no_grad() if self.freeze_action_encoder else torch.enable_grad():
            if self.use_action_vae:
                action_posterior, action_latents = self.action_encoder(gt_actions, deterministic=not self.training)
            else:
                action_latents = self.action_encoder(gt_actions)

        flow_loss, metrics = self.FM.compute_loss(
            self.flow_net,
            target=action_latents,
            start=state_latents,
            global_cond=visual_latents
        )
        loss = flow_loss
        metrics["flow_loss"] = flow_loss.item()

        if self.enc_contrastive_weight > 0:
            contrastive_loss = compute_contrastive_loss(visual_latents, action_latents)
            loss += self.enc_contrastive_weight * contrastive_loss
            metrics["enc_contrastive_loss"] = contrastive_loss.item()

        if self.freeze_action_encoder and self.freeze_action_decoder:
            return loss, metrics

        if not self.freeze_action_encoder and self.use_action_vae and self.action_kl_weight > 0:
            action_kl_loss = action_posterior.kl().mean()
            metrics["action_kl_loss"] = action_kl_loss.item()
            loss += self.action_kl_weight * action_kl_loss

        if self.decode_flow_latents and not self.freeze_action_encoder and not self.freeze_action_decoder:
            action_latents_pred = self.FM.sample(
                self.flow_net,
                shape=(batch_size, self.latent_dim),
                device=state_latents.device,
                start=state_latents,
                num_steps=self.num_sampling_steps,
                global_cond=visual_latents,
            )

            if self.consistency_weight > 0:
                consistency_loss = F.mse_loss(action_latents_pred, action_latents)
                loss += self.consistency_weight * consistency_loss
                metrics["consistency_loss"] = consistency_loss.item()

            if self.flow_contrastive_weight > 0:
                contrastive_loss = compute_contrastive_loss(visual_latents, action_latents_pred)
                loss += self.flow_contrastive_weight * contrastive_loss
                metrics["flow_contrastive_loss"] = contrastive_loss.item()

            if self.flow_action_recon_weight > 0 and not self.freeze_action_decoder:
                actions_recon = self.action_decoder(action_latents_pred)
                action_recon_loss = self.recon_loss_fn(actions_recon, gt_actions)
                metrics["flow_action_recon_loss"] = action_recon_loss.item()
                loss += self.flow_action_recon_weight * action_recon_loss
        else:
            action_latents_pred = action_latents

        if self.enc_action_recon_weight > 0 and not self.freeze_action_decoder:
            actions_recon = self.action_decoder(action_latents)
            action_recon_loss = self.recon_loss_fn(actions_recon, gt_actions)
            metrics["enc_action_recon_loss"] = action_recon_loss.item()
            loss += self.enc_action_recon_weight * action_recon_loss

        return loss, metrics

    def generate_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = batch[self.config.task.state_key].shape[0]
        state_latents = self._encode_state(batch)
        obs_features = self.observer(batch)
        visual_latents = self.obs_encoder(obs_features)

        action_latents_pred = self.FM.sample(
            self.flow_net,
            (batch_size, self.latent_dim),
            state_latents.device,
            self.num_sampling_steps,
            start=state_latents,
            global_cond=visual_latents,
            return_traces=False,
        )

        with torch.no_grad() if self.freeze_action_decoder else torch.enable_grad():
            actions_pred = self.action_decoder(action_latents_pred)

        return actions_pred

    @torch.no_grad
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        input_keys = self.config.task.image_keys + [self.config.task.state_key]
        batch = {k: v.unsqueeze(1) for k, v in batch.items() if k in input_keys}
        batch = self.normalize_inputs(batch)
        if len(self._action_queue) == 0:
            actions = self.generate_actions(batch)
            actions = actions[:, : self.action_horizon]
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config.optimizer_lr,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
            weight_decay=self.config.optimizer_weight_decay,
        )

    def get_scheduler(
        self, optimizer: torch.optim.Optimizer, num_training_steps: int
    ) -> torch.optim.lr_scheduler.LambdaLR | None:
        return get_scheduler(
            name=self.config.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=self.config.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def visualize(self, batch: dict[str, torch.Tensor], num_samples: int = 1) -> dict[str, plt.Figure]:
        self.eval()
        for key in batch:
            batch[key] = batch[key][:num_samples]
        batch = self.normalize_inputs(batch)
        device = batch[self.config.task.action_key].device

        with torch.no_grad():
            pred_norm = self.generate_actions(batch)
            pred = self.unnormalize_outputs({"action": pred_norm})["action"]
        gt = batch[self.config.task.action_key]

        state_latents = self._encode_state(batch)
        obs_feats = self.observer(batch)
        visual_latents = self.obs_encoder(obs_feats)
        _, (latents_hist, _) = self.FM.sample(
            self.flow_net,
            shape=(num_samples, self.latent_dim),
            device=device,
            num_steps=self.num_sampling_steps,
            start=state_latents,
            global_cond=visual_latents,
            return_traces=True,
        )

        viz: dict[str, plt.Figure] = {}
        for i in range(num_samples):
            fig1, ax1 = plt.subplots()
            traj_pred = pred[i, :, :2].cpu().numpy()
            traj_gt = gt[i, :, :2].cpu().numpy()
            plot_trajectory(ax=ax1, pred=traj_pred, target=traj_gt)
            viz[f"cmp_{i}"] = fig1

            traj_actions = []
            for lh in latents_hist:
                lat_i = lh[i].to(device)
                act_traj = self.action_decoder(lat_i.unsqueeze(0)).squeeze(0).cpu().numpy()
                traj_actions.append(act_traj)

            fig2 = plot_ode_steps(traj_actions)
            viz[f"denoise_{i}"] = fig2

        return viz


def compute_contrastive_loss(state_features, action_features, temperature=0.07):
    batch_size = state_features.size(0)
    state_features = F.normalize(state_features, dim=1)
    action_features = F.normalize(action_features, dim=1)

    logits = torch.matmul(state_features, action_features.T) / temperature

    labels = torch.arange(batch_size, device=logits.device)
    loss_s2a = F.cross_entropy(logits, labels)
    loss_a2s = F.cross_entropy(logits.T, labels)

    return (loss_s2a + loss_a2s) / 2
