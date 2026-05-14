import torch

from flare.flow.base_flow_matcher import BaseFlowMatcher


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss.
    """
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)))
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()


def adaptive_imf_loss(error, norm_p=1.0, norm_eps=0.01):
    per_sample_loss = torch.sum(error ** 2, dim=tuple(range(1, error.ndim)))
    adaptive_weight = (per_sample_loss + norm_eps).pow(norm_p)
    return (per_sample_loss / stopgrad(adaptive_weight)).mean()


def dispersive_loss(z, tau=1.0):
    """
    Dispersive Loss.
    """
    if z.shape[0] <= 1:
        return 0.0
    dist_matrix = torch.cdist(z, z, p=2) ** 2
    # Normalize to prevent overflow/underflow
    dist_matrix = dist_matrix / (torch.max(dist_matrix).detach() + 1e-8)
    exp_term = torch.exp(-dist_matrix / tau)
    mean_exp = torch.mean(exp_term)
    loss = torch.log(mean_exp)
    return loss


class MeanFlowMatcher(BaseFlowMatcher):
    def __init__(
        self,
        flow_ratio=0.5,
        time_dist_mu=-0.4,
        time_dist_sigma=1.0,
        adaptive_loss_gamma=0.5,
        norm_p=1.0,
        norm_eps=0.01,
        aux_v_loss_weight=1.0,
        dispersive_loss_tau=1.0,
        dispersive_loss_weight=0.0,
        cfg_scale=0.5,
        use_imf=False,
        **kwargs,
    ):
        super().__init__()
        self.flow_ratio = flow_ratio
        self.time_dist_mu = time_dist_mu
        self.time_dist_sigma = time_dist_sigma
        self.adaptive_loss_gamma = adaptive_loss_gamma
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.aux_v_loss_weight = aux_v_loss_weight
        self.dispersive_loss_tau = dispersive_loss_tau
        self.dispersive_loss_weight = dispersive_loss_weight
        self.cfg_scale = cfg_scale
        self.use_imf = use_imf

    def _unpack_output(self, output, require_aux_v=False):
        if not isinstance(output, tuple):
            return output, None, None

        if len(output) == 2:
            prediction, internal_features = output
            if require_aux_v:
                raise ValueError("Improved MeanFlow requires the model to return (u, v, internal_features).")
            return prediction, None, internal_features

        if len(output) == 3:
            u, v, internal_features = output
            return u, v, internal_features

        raise ValueError(f"Unexpected MeanFlow model output with {len(output)} values.")

    def sample_t_r(self, batch_size, device):
        """
        Samples t and r from a log-normal distribution.
        """
        # Log-normal distribution
        normal_samples = (
            torch.randn(batch_size, 2, device=device) * self.time_dist_sigma
            + self.time_dist_mu
        )
        samples = torch.sigmoid(normal_samples)

        # t = max, r = min
        t = torch.max(samples, dim=1)[0]
        r = torch.min(samples, dim=1)[0]

        # Set r=t for a portion of the batch
        num_selected = int(self.flow_ratio * batch_size)
        indices = torch.randperm(batch_size, device=device)[:num_selected]
        r[indices] = t[indices]

        return t, r

    def compute_loss(self, model, target, start=None, **kwargs):
        """
        Compute the MeanFlow/iMF loss.
        iMF assumes `model` returns (u, v, internal_features).
        """
        if start is None:
            raise ValueError("MeanFlowMatcher requires a 'start' (vision latent) tensor.")

        x1 = target
        x0 = start
        batch_size = x0.shape[0]
        device = x0.device

        # Sample t and r
        t, r = self.sample_t_r(batch_size, device)
        t_ = t.view(-1, *([1] * (x0.dim() - 1)))
        h = t - r
        h_ = h.view(-1, *([1] * (x0.dim() - 1)))

        # Define path and sample z_t
        z_t = (1 - t_) * x1 + t_ * x0

        # Ground-truth instantaneous velocity v = dx/dt
        v = x0 - x1

        def pred_meanflow(z_in, t_in, r_in):
            return model(x=z_in, timestep=t_in, h=t_in - r_in, **kwargs)

        if self.use_imf:
            with torch.no_grad():
                _, v_net, _ = self._unpack_output(pred_meanflow(z_t, t, t), require_aux_v=True)
            dz_tangent = v_net

            def pred_imf(z_in, t_in, r_in):
                u, v_pred, _ = self._unpack_output(pred_meanflow(z_in, t_in, r_in), require_aux_v=True)
                return u, v_pred

            (predicted_mean_vel, predicted_v), (dudt, _) = torch.autograd.functional.jvp(
                pred_imf,
                (z_t, t, r),
                (dz_tangent, torch.ones_like(t), torch.zeros_like(r)),
                create_graph=True,
            )

            compound_velocity = predicted_mean_vel + h_ * stopgrad(dudt)
            imf_loss = adaptive_imf_loss(compound_velocity - stopgrad(v), norm_p=self.norm_p, norm_eps=self.norm_eps)
            aux_v_loss = adaptive_imf_loss(predicted_v - stopgrad(v), norm_p=self.norm_p, norm_eps=self.norm_eps)

            loss = imf_loss + self.aux_v_loss_weight * aux_v_loss
            metrics = {
                'imf_loss': imf_loss.item(),
                'aux_v_loss': aux_v_loss.item(),
                'imf_mse': torch.mean((compound_velocity - v) ** 2).item(),
                'aux_v_mse': torch.mean((predicted_v - v) ** 2).item(),
            }
        else:
            dz_tangent = v

            def pred_meanflow_u(z_in, t_in, r_in):
                u, _, _ = self._unpack_output(pred_meanflow(z_in, t_in, r_in))
                return u

            predicted_mean_vel, dudt = torch.autograd.functional.jvp(
                pred_meanflow_u,
                (z_t, t, r),
                (dz_tangent, torch.ones_like(t), torch.zeros_like(r)),
                create_graph=True,
            )

            u_tgt = v - h_ * dudt
            error = predicted_mean_vel - stopgrad(u_tgt)
            meanflow_loss = adaptive_l2_loss(error, gamma=self.adaptive_loss_gamma)

            loss = meanflow_loss
            metrics = {'meanflow_loss': meanflow_loss.item()}

        _, _, internal_features = self._unpack_output(pred_meanflow(z_t, t, r), require_aux_v=self.use_imf)

        # Dispersive Loss
        if self.dispersive_loss_weight > 0 and internal_features is not None:
            dis_loss_total = 0.0
            # internal_features is a list of tensors from the network's hidden layers
            for features in internal_features:
                dis_loss_total += dispersive_loss(features, tau=self.dispersive_loss_tau)

            metrics['dispersive_loss'] = dis_loss_total.item()
            loss += self.dispersive_loss_weight * dis_loss_total

        metrics['loss'] = loss.item()
        return loss, metrics

    def sample(self, model, shape, device, num_steps=None, return_traces=False, start=None, **kwargs):
        """
        Generate samples in 1-NFE using MeanFlow.
        """
        if start is None:
            raise ValueError("MeanFlowMatcher requires a 'start' (vision latent) tensor for sampling.")

        x_source = start
        batch_size = x_source.shape[0]

        if num_steps is None:
            num_steps = self.num_sampling_steps

        x = x_source
        if return_traces:
            traj_history = [x.detach().clone().cpu()]
            vel_history = [torch.zeros_like(x).cpu()]

        for step in range(num_steps):
            t_scalar = 1.0 - step / num_steps
            r_scalar = 1.0 - (step + 1) / num_steps
            t = torch.full((batch_size,), t_scalar, device=device)
            h = torch.full((batch_size,), t_scalar - r_scalar, device=device)
            mean_velocity, _, _ = self._unpack_output(model(x=x, timestep=t, h=h, **kwargs))
            x = x - h.view(-1, *([1] * (x.dim() - 1))) * mean_velocity

            if return_traces:
                traj_history.append(x.detach().clone().cpu())
                vel_history.append(mean_velocity.detach().clone().cpu())

        if return_traces:
            return x, (traj_history, vel_history)

        return x
