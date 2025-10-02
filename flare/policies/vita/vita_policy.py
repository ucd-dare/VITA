import torch
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers.optimization import get_scheduler

from flare.factory import registry
from flare.flow.flow_matchers import get_flow_matcher
from flare.policies.observers.resnet_observer import ResNetObserver
from flare.utils.normalize import Normalize, Unnormalize

from flare.networks.vita.action_ae import get_autoencoder
from flare.networks.vita.action_vae import get_variational_autoencoder

from flare.policies import BasePolicy
from flare.policies.vita.simple_flow_net import SimpleFlowNet
from flare.visualizer.visualizer import plot_trajectory, plot_ode_steps

logger = logging.getLogger(__name__)


@registry.register_policy("vita")
class VitaPolicy(BasePolicy):
    def __init__(self, config, stats):
        super().__init__(config, stats)

        self.config = config
        self.stats = stats
        self.num_sampling_steps = config.policy.flow_matcher.num_sampling_steps
        self.action_horizon = config.policy.action_horizon
        self.action_dim = config.task.action_dim
        self.obs_horizon = config.policy.obs_horizon

        self.normalize_inputs = Normalize(config.task.image_keys+[config.task.state_key], stats)
        self.normalize_targets = Normalize([config.task.action_key], stats)
        self.unnormalize_outputs = Unnormalize([config.task.action_key], stats)
        self._action_queue = None

        self.observer = ResNetObserver(
            state_key=config.task.state_key,
            image_keys=config.task.image_keys,
            resize_shape=config.resize_shape,
            crop_shape=config.crop_shape,
            state_dim=config.task.state_dim,
            tokenize=config.policy.observer.tokenize,
        )
        if config.policy.observer.tokenize:
            self.obs_dim = 512
        else:
            self.obs_dim = len(self.config.task.image_keys) * 512 + self.config.task.state_dim

        self.FM = get_flow_matcher(**config.policy.flow_matcher)

        action_ae_net_config = config.policy.action_ae.net
        self.use_action_vae = config.policy.action_ae.use_variational
        self.latent_dim = action_ae_net_config.latent_dim
        self.num_sampling_steps = config.policy.flow_matcher.num_sampling_steps
        self.freeze_action_encoder = config.policy.action_ae.freeze_encoder
        self.freeze_action_decoder = config.policy.action_ae.freeze_decoder
        self.flow_action_recon_weight = config.policy.action_ae.flow_recon_weight
        self.enc_action_recon_weight = config.policy.action_ae.enc_recon_weight

        recon_loss_type = config.policy.action_ae.recon_loss_type
        self.action_kl_weight = config.policy.action_ae.kl_weight
        if recon_loss_type == 'l1':
            self.recon_loss_fn = F.l1_loss
        elif recon_loss_type == 'l2':
            self.recon_loss_fn = F.mse_loss
        else:
            raise ValueError(f"Unsupported recon_loss_type: {recon_loss_type}. Use 'l1' or 'l2'.")

        self.use_vision_token = False  # TODO: use vision tokens
        self.use_obs_vae = config.policy.obs_ae.use_variational
        self.obs_kl_weight = config.policy.obs_ae.kl_weight
        self.enc_contrastive_weight = config.policy.vita.enc_contrastive_weight
        self.flow_contrastive_weight = config.policy.vita.flow_contrastive_weight

        logger.info("Using 1D vision features.")
        self.obs_encoder = torch.nn.Linear(self.obs_dim, self.latent_dim)

        # Initialize action encoder/decoder
        self._init_action_vae(action_ae_net_config)

        # Initialize flow network
        logger.info("Using MLP for flow velocity prediction.")
        self.flow_net = SimpleFlowNet(
            input_dim=self.latent_dim,
            hidden_dim=config.policy.flow_net.hidden_dim,
            output_dim=self.latent_dim,
            num_layers=config.policy.flow_net.num_layers,
            num_heads=config.policy.flow_net.num_heads,
            mlp_ratio=config.policy.flow_net.get('mlp_ratio', 4.0),
            dropout=config.policy.flow_net.get('dropout', 0.0),
        )

        self.reset()

    def _init_action_vae(self, action_ae_net_config):
        """Initialize action encoder and decoder."""
        encoder_type = action_ae_net_config.encoder_type
        decoder_type = action_ae_net_config.decoder_type

        logger.info(f"Using {encoder_type} encoder and {decoder_type} decoder for action autoencoder.")
        if self.use_action_vae:
            self.action_encoder, self.action_decoder = get_variational_autoencoder(self.config, self.config.policy)
        else:
            self.action_encoder, self.action_decoder = get_autoencoder(self.config, self.config.policy)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        obs_features = self.observer(batch)
        if self.use_obs_vae:
            obs_posterior, obs_latents = self.obs_encoder(obs_features, deterministic=not self.training)
        else:
            obs_latents = self.obs_encoder(obs_features)
            obs_posterior = None

        batch_size = obs_features.shape[0]
        gt_actions = batch[self.config.task.action_key]

        with torch.no_grad() if self.freeze_action_encoder else torch.enable_grad():
            if self.use_action_vae:
                action_posterior, action_latents = self.action_encoder(gt_actions, deterministic=not self.training)
            else:
                action_latents = self.action_encoder(gt_actions)

        # Flow matching loss: obs_latents -> action_latents
        flow_loss, metrics = self.FM.compute_loss(
            self.flow_net,
            target=action_latents,
            start=obs_latents # Use visual latents as the flow source
        )
        loss = flow_loss
        metrics['flow_loss'] = flow_loss.item()

        # Observation VAE losses
        if obs_posterior is not None and self.obs_kl_weight > 0:
            obs_kl_loss = obs_posterior.kl().mean()
            loss += self.obs_kl_weight * obs_kl_loss
            metrics['obs_kl_loss'] = obs_kl_loss.item()

        if self.enc_contrastive_weight > 0:
            image_features = obs_latents.view(batch_size, -1)
            action_features = action_latents.view(batch_size, -1)
            contrastive_loss = compute_contrastive_loss(image_features, action_features)
            loss += self.enc_contrastive_weight * contrastive_loss
            metrics['enc_contrastive_loss'] = contrastive_loss.item()

        # Skip action VAE losses if freezing
        if self.freeze_action_encoder and self.freeze_action_decoder:
            return loss, metrics

        # Action VAE losses
        if not self.freeze_action_encoder and self.use_action_vae and self.action_kl_weight > 0:
            # Use built-in KL divergence from DiagonalGaussianDistribution
            action_kl_loss = action_posterior.kl().mean()
            metrics['action_kl_loss'] = action_kl_loss.item()
            loss += self.action_kl_weight * action_kl_loss

        # Sample action latents and decode for reconstruction losses
        if self.config.policy.vita.decode_flow_latents and not self.freeze_action_encoder and not self.freeze_action_decoder:
            action_latents_pred = self.FM.sample(
                self.flow_net,
                shape=(batch_size, self.latent_dim),
                device=obs_latents.device,
                start=obs_latents, # Use visual latents as the flow source
                num_steps=self.num_sampling_steps
            )

            if self.config.policy.vita.consistency_weight > 0:
                consistency_loss = F.mse_loss(action_latents_pred, action_latents)
                loss += self.config.policy.vita.consistency_weight * consistency_loss
                metrics['consistency_loss'] = consistency_loss.item()

            if self.flow_contrastive_weight > 0:
                image_features = obs_latents.view(batch_size, -1)
                action_features = action_latents_pred.view(batch_size, -1)
                contrastive_loss = compute_contrastive_loss(image_features, action_features)
                loss += self.flow_contrastive_weight * contrastive_loss
                metrics['flow_contrastive_loss'] = contrastive_loss.item()

            if self.flow_action_recon_weight > 0 and not self.freeze_action_decoder:
                actions_recon = self.action_decoder(action_latents_pred)
                action_recon_loss = self.recon_loss_fn(actions_recon, gt_actions)
                metrics['flow_action_recon_loss'] = action_recon_loss.item()
                loss += self.flow_action_recon_weight * action_recon_loss
        else:
            action_latents_pred = action_latents

        # Encoder reconstruction losses
        if self.enc_action_recon_weight > 0 and not self.freeze_action_decoder:
            actions_recon = self.action_decoder(action_latents)
            action_recon_loss = self.recon_loss_fn(actions_recon, gt_actions)
            metrics['enc_action_recon_loss'] = action_recon_loss.item()
            loss += self.enc_action_recon_weight * action_recon_loss

        return loss, metrics

    def generate_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = batch[self.config.task.state_key].shape[0]
        obs_features = self.observer(batch)

        if self.use_obs_vae:
            obs_posterior, obs_latents = self.obs_encoder(obs_features, deterministic=True)
        else:
            obs_latents = self.obs_encoder(obs_features)

        action_latents_pred = self.FM.sample(
            self.flow_net,
            (batch_size, self.latent_dim),
            obs_latents.device,
            self.num_sampling_steps,
            start=obs_latents, # Use visual latents as the flow source
            return_traces=False
        )

        with torch.no_grad() if self.freeze_action_decoder else torch.enable_grad():
            actions_pred = self.action_decoder(action_latents_pred)

        return actions_pred

    @torch.no_grad
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        batch = {k: v.unsqueeze(1) for k, v in batch.items() if k in self.config.task.image_keys + [self.config.task.state_key]}
        batch = self.normalize_inputs(batch)
        if len(self._action_queue) == 0:
            actions = self.generate_actions(batch)
            actions = actions[:, :self.action_horizon]
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config.optimizer_lr,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
            weight_decay=self.config.optimizer_weight_decay
        )

    def get_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int) -> torch.optim.lr_scheduler.LambdaLR | None:
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

        obs_feats = self.observer(batch)
        if self.config.policy.obs_ae.use_variational:
            obs_posterior, obs_latents = self.obs_encoder(obs_feats, deterministic=True)
        else:
            obs_latents = self.obs_encoder(obs_feats)

        action_latents, (latents_hist, vel_hist) = self.FM.sample(
            self.flow_net,
            shape=(num_samples, self.latent_dim),
            device=device,
            num_steps=self.num_sampling_steps,
            start=obs_latents, # Use visual latents as the flow source
            return_traces=True
        )

        viz: dict[str, plt.Figure] = {}
        for i in range(num_samples):
            # --- Figure 1: GT vs Pred ---
            fig1, ax1 = plt.subplots()
            traj_pred = pred[i, :, :2].cpu().numpy()
            traj_gt = gt[i, :, :2].cpu().numpy()
            plot_trajectory(ax=ax1, pred=traj_pred, target=traj_gt)
            viz[f"cmp_{i}"] = fig1

            # --- Figure 2: Denoising Steps ---
            traj_actions = []
            for lh in latents_hist:
                lat_i = lh[i]
                lat_i = lat_i.to(device)
                act_traj = self.action_decoder(lat_i.unsqueeze(0)).squeeze(0).cpu().numpy()
                traj_actions.append(act_traj)

            fig2 = plot_ode_steps(traj_actions)
            viz[f"denoise_{i}"] = fig2

        return viz


def compute_contrastive_loss(image_features, action_features, temperature=0.07):
    # Contrastive loss between image and action feautres (InfoNCE)
    # Can provide an additional boost on top of FLD and FLC

    # Normalize features
    batch_size = image_features.size(0)
    image_features = F.normalize(image_features, dim=1)
    action_features = F.normalize(action_features, dim=1)

    # Compute similarity matrix
    logits = torch.matmul(image_features, action_features.T) / temperature

    # Symmetric contrastive loss (image-to-action + action-to-image)
    labels = torch.arange(batch_size, device=logits.device)
    loss_i2a = F.cross_entropy(logits, labels)
    loss_a2i = F.cross_entropy(logits.T, labels)

    return (loss_i2a + loss_a2i) / 2
