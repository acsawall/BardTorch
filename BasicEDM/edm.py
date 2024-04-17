# EDM from algorithms discussed in https://arxiv.org/pdf/2206.00364.pdf
import copy

import torch
import torch.nn as nn
import numpy as np


# Algorithm 2 stochastic sampler https://github.com/NVlabs/edm/blob/main/generate.py#L25
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, use_ema=True,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, labels=class_labels, use_ema=use_ema).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, labels=class_labels, use_ema=use_ema).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# Elucidated Diffusion Model implementation
class EDiffusion(nn.Module):
    def __init__(
            self,
            model=None,
            device=torch.device("cpu")
    ):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)

        # parameters
        self.sigma_data = 0.5       # default 0.5
        self.sigma_max = 80         # default 80
        self.sigma_min = 0.002      # default 0.002
        self.rho = 7                # default 7

        # constants
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5

        # exponential moving avg
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.ema_rampup_ratio = 0.05
        self.ema_halflife_k_img = 500

    # Proposed forward function for preconditioning
    # https://github.com/NVlabs/edm/blob/main/training/networks.py#L654
    def get_denoiser(self, x, sigma, use_ema=False, **kwargs):
        #x = x.to(torch.float32)
        #sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        sigma[sigma == 0] = self.sigma_min      # set all zero points to sigma_min
        labels = kwargs["labels"] if "labels" in kwargs else None
        #labels = None if self.n_classes == 0 else torch.zeros([1, self.n_classes],device=x.device) if labels is None \
            #else labels.to(torch.float32).reshape(-1, self.n_classes)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        if use_ema:
            model_out = self.ema(torch.einsum('b,bijk->bijk', c_in, x), c_noise, class_labels=labels)
        else:
            model_out = self.model(torch.einsum('b,bijk->bijk', c_in, x), c_noise, class_labels=labels)

        try:
            model_out = model_out.sample
        except Exception:
            pass

        return torch.einsum('b,bijk->bijk', c_skip, x) + torch.einsum('b,bijk->bijk', c_out, model_out)

    def train_one_step(self, images, labels=None, augment_pipe=None, **kwargs):
        rnd_normal = torch.randn([images.shape[0]], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / ((sigma * self.sigma_data) ** 2)

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        noise = torch.randn_like(y)
        n = torch.einsum('b,bijk->bijk', sigma, noise)
        D_yn = self.get_denoiser(y + n, sigma, labels=labels, augment_labels=augment_labels)
        loss = torch.einsum('b,bijk->bijk', weight, ((D_yn - y) ** 2))
        return loss.mean()

    def __call__(self, x, sigma, labels=None, augment_labels=None, use_ema=True):
        if sigma.shape == torch.Size([]):
            sigma = sigma * torch.ones([x.shape[0]]).to(x.device)
        return self.get_denoiser(x.float(), sigma.float(), use_ema=use_ema, labels=labels, augment_labels=augment_labels)

    def update_exp_moving_avg(self, step, batch_size):
        ema_halflife_n_img = self.ema_halflife_k_img * 1000
        if self.ema_rampup_ratio is not None:
            ema_halflife_n_img = min(ema_halflife_n_img, step * batch_size * self.ema_rampup_ratio)
        ema_B = 0.5 ** (batch_size / max(ema_halflife_n_img, 1e-8))
        for p_ema, p_net in zip(self.ema.parameters(), self.model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_B))

    # Helper method for edm_sampler, same as in NVIDIA's precondition
    # https://github.com/NVlabs/edm/blob/main/training/networks.py#L670
    #@staticmethod
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
