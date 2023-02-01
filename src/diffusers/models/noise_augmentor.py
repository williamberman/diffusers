import math
from typing import List, Optional, Union

import numpy as np
import torch
from torch import nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import randn_tensor
from .embeddings import Timesteps
from .modeling_utils import ModelMixin


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


class NoiseAugmentor(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        embedding_dim: int = 768,
        max_noise_level: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        set_alpha_to_one: bool = True,
    ):
        super().__init__()
        self.max_noise_level = max_noise_level

        self.mean = nn.Parameter(torch.zeros(1, embedding_dim))
        self.std = nn.Parameter(torch.ones(1, embedding_dim))

        self.time_embed = Timesteps(embedding_dim, True, 0)

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, max_noise_level, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, max_noise_level, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(max_noise_level)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

    def scale(self, x):
        x = (x - self.mean) * 1.0 / self.std
        return x

    def unscale(self, x):
        x = (x * self.std) + self.mean
        return x

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def forward(
        self,
        x: torch.Tensor,
        noise_level: int,
        noise: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        noise_level = torch.tensor([noise_level] * x.shape[0], device=x.device)

        if noise is None:
            noise = randn_tensor(x.shape, generator=generator, device=self.device, dtype=x.dtype)

        x = self.scale(x)
        x = self.add_noise(original_samples=x, timesteps=noise_level, noise=noise)
        x = self.unscale(x)

        noise_level = self.time_embed(noise_level)

        # `self.time_embed` does not contain any weights and will always return f32 tensors,
        # but we might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        noise_level = noise_level.to(self.dtype)

        combined = torch.cat((x, noise_level), 1)

        return combined
