import torch
import numpy as np

class DiagonalGaussianDistribution():
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = torch.zeros_like(self.mean)
            self.std = torch.zeros_like(self.mean)

    def sample(self):
        if self.deterministic:
            return self.mean
        else:
            eps = torch.randn_like(self.mean)
            return self.mean + self.std * eps

    def kl(self, other=None):
        dims = np.arange(1, self.mean.dim()).tolist()
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device)
        if other is None:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=dims
            )
        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
            dim=dims
        )

    def nll(self, sample):
        dims = np.arange(1, sample.dim()).tolist()
        if self.deterministic:
            return torch.zeros(sample.shape[0], device=sample.device)
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims
        )

    def mode(self):
        return self.mean