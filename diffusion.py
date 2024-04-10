import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Optional

from einops import rearrange, reduce
from math import sqrt


# A callable Sigma Distribution function for applying noise
class SigmaDistribution:
    # P_mean = -1.2
    # P_std = 1.0
    def __init__(self, mean: float = -1.2, std: float = 1.0):
        self.mean = mean  # -3.0
        self.std = std

    def __call__(self, num_samples: int, device: torch.device):
        normal = self.mean + self.std * torch.randn((num_samples,), device=device)
        return normal.exp()


# Primarily based on method described in: https://arxiv.org/pdf/2206.00364.pdf
# Some helper functions and structure from Zach Evan's implementation
#   at: https://github.com/Harmonai-org/audio-diffusion-pytorch-fork/tree/main
class EDiffusion(nn.Module):
    def __init__(
            self,
            net: nn.Module,
            sigma_distribution: SigmaDistribution,
            sigma_data: float,  # Standard deviation of some data
            device=torch.device("cuda"),
            dynamic_threshold: float = 0.0
    ):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution
        self.sigma_data = sigma_data
        self.device = device
        self.dynamic_threshold = dynamic_threshold

    """
    Returns scaling variables c_skip, c_out, c_in, c_noise
    
    Return type - Tuple[Tensor, ...]
    """
    def get_scaling(self, sigmas: Tensor):
        c_noise = torch.log(sigmas) * 0.25
        sigmas = rearrange(sigmas, "b -> b 1 1")
        c_skip = (self.sigma_data ** 2) / (sigmas ** 2 + self.sigma_data ** 2)
        c_out = sigmas * self.sigma_data / sqrt(self.sigma_data ** 2 + sigmas ** 2)
        c_in = 1 / sqrt(sigmas ** 2 + self.sigma_data ** 2)
        return c_skip, c_out, c_in, c_noise

    """
    Return the loss weighting λ(σ)
    """
    def get_loss_weighting(self, sigmas: Tensor):
        return (sigmas ** 2 + self.sigma_data ** 2) / ((sigmas * self.sigma_data) ** 2)

    """
    Returns the denoiser function, notated in the paper as D_Θ(x; σ)
    """
    def denoiser_fn(
            self,
            x: Tensor,
            sigmas: Optional[Tensor] = None,
            sigma: Optional[Tensor] = None,
            **kwargs
    ):
        batch_size = x.shape[0]
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=self.device)

        # Get scaling
        c_skip, c_out, c_in, c_noise = self.get_scaling(sigmas)
        # F_theta(c_in * x; c_noise)
        x_prediction = self.net(c_in * x, c_noise, **kwargs)
        x_denoiser = c_skip * x + c_out * x_prediction
        return clip(x_denoiser, self.dynamic_threshold)

    """
    Forward Step
    """
    def forward(
            self,
            x: Tensor,
            noise: Tensor = None,
            **kwargs
    ):
        batch_size = x.shape[0]

        # Get sigma distribution to apply noise
        sigma_dist = self.sigma_distribution(num_samples=batch_size, device=self.device)
        padded_dist = rearrange(sigma_dist, "b -> b 1 1")

        # Apply noise to x
        if noise is None:
            noise = torch.randn_like(x)
        x_noised = x + padded_dist * noise

        # Apply denoiser function
        x_denoised = self.denoiser_fn(x=x_noised, sigmas=sigma_dist, **kwargs)

        # Compute loss
        loss = F.mse_loss(x_denoised, x, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")
        loss = loss * self.get_loss_weighting(sigma_dist)
        mean_loss = loss.mean()
        return mean_loss


def to_batch(batch_size, device, x=None, xs=None):
    if x is not None and xs is None:
        xs = torch.full(size=(batch_size,), fill_value=x).to(device)
    return xs


def pad_dims(x: Tensor, ndim: int) -> Tensor:
    # Pads additional ndims to the right of the tensor
    return x.view(*x.shape, *((1,) * ndim))


# Helper function to clamp each batch to the passed dynamic threshold
def clip(x: Tensor, threshold: float = 0.0):
    if threshold == 0.0:
        return x.clamp(-1.0, 1.0)
    else:
        x_flat = rearrange(x, "b ... -> b (...)")
        scale = torch.quantile(x_flat.abs(), threshold, dim=-1)
        scale.clamp_(min=1.0)
        scale = pad_dims(scale, ndim=x.ndim - scale.ndim)
        x = x.clamp(-scale, scale) / scale
        return x
