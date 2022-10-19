from dataclasses import dataclass
from typing import Tuple, Union

import torch
import numpy as np

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin

def gumbel_noised(logits):
    uniform = torch.rand_like(logits, device=logits.device)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    noised = gumbel_noise + logits
    return noised


def alpha_schedules(num_diffusion_timesteps, att_1 = 0.99999, att_T = 0.000009):
    att = np.arange(0, num_diffusion_timesteps)/(num_diffusion_timesteps-1)*(att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:]/att[:-1]
    att = np.concatenate((att[1:], [1]))
    return at, att


def gamma_schedules(time_step, ctt_1 = 0.000009, ctt_T = 0.99999):
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct
    ctt = np.concatenate((ctt[1:], [0]))
    return ct, ctt


@dataclass
class VQDiffusionSchedulerOutput(BaseOutput):
    x_t_min_1: torch.LongTensor


class VQDiffusionScheduler(SchedulerMixin, ConfigMixin):
    """
    TODO

    For more details, see the original paper: https://arxiv.org/abs/2111.14822


    Args:
        TODO
    """
    @register_to_config
    def __init__(
        self,
        # number of classes, including the mask
        num_embed: int,
        num_train_timesteps: int = 100,
        min_logged_value: float = -70.0,
    ):
        self.num_embed = num_embed
        self.min_logged_value = min_logged_value

        # By convention, the index for the mask class is the last class index
        self.mask_class = self.num_embed - 1

        at, att = alpha_schedules(num_train_timesteps)
        ct, ctt = gamma_schedules(num_train_timesteps)

        num_non_mask_classes = self.num_embed - 1
        bt = (1-at-ct) / num_non_mask_classes
        btt = (1-att-ctt) / num_non_mask_classes

        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)

        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        self.log_at = log_at.float()
        self.log_bt = log_bt.float()
        self.log_ct = log_ct.float()
        self.log_cumprod_at = log_cumprod_at.float()
        self.log_cumprod_bt = log_cumprod_bt.float()
        self.log_cumprod_ct = log_cumprod_ct.float()

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())


    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps
        timesteps = np.arange(0, self.num_inference_steps)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).to(device)


    def step(
        self, 
        log_p_x_0, 
        x_t, 
        t,
        return_dict: bool = True,
    ) -> Union[VQDiffusionSchedulerOutput, Tuple]:
        """
        log_x_start is the log probs of the unnoised image probabilities. probabilities as col vectors
        (batch, )

        i.e. log_x_start[batch, class probability, latent pixel]
        """
        # TODO remove
        torch.save(log_p_x_0, f"/content/diffusers-out/log_p_x_0-{t[0]}.pt")

        log_x_t_min_1 = self.q_posterior(log_p_x_0=log_p_x_0, x_t=x_t, t=t)

        log_x_t_min_1 = gumbel_noised(log_x_t_min_1)

        x_t_min_1 = log_x_t_min_1.argmax(1)

        if not return_dict:
            return (x_t_min_1,)

        num_masked = (x_t_min_1 == self.mask_class).count_nonzero()
        print(f"num masked {num_masked}")

        # TODO remove
        torch.save(x_t_min_1, f"/content/diffusers-out/x_t_min_1-{t[0]}.pt")

        return VQDiffusionSchedulerOutput(x_t_min_1=x_t_min_1)


    def q_posterior(self, *, log_p_x_0, x_t, t):
        """
        Calculates Equation 11 in log space

        p(x_{t-1} | x_t, y) = sum( q(x_{t-1} | x_t, x_0') p(x_0' | x_t, y) )

        Where: 
        - The sum is over the predicted classes for the denoised image.
        - Writing \tilde{x}_{0} as x_0'.
            - x_0' is the noiseless token distribution predicted by the transformer. 
        - Writing p_{\theta} as p

        Args:
            log_x_0 (`torch.FloatTensor` of shape `(batch_size, num classes, num latent pixels)`): 
                The log probabilities of the latent pixel classes at time step 0

            x_t (`torch.FloatTensor` of shape `(batch_size, num latent pixels)`): 
                The predictions of the latent pixel classes at time steps `t`

            t (`torch.LongTensor` of shape `(batch_size,)`):
                Current diffusion steps

        Returns:
            `torch.FloatTensor` of shape `(batch_size, log probability of class, latent pixel)`:
                The log probabilities of the latent pixel classes at time step `t-1`.
        """
        device = log_p_x_0.device
        bsz, _, num_latent_pixels  = log_p_x_0.shape

        log_p_x_t_min_1_given_x_t = torch.empty((bsz, self.num_embed, num_latent_pixels), device=device)

        # TODO remove batch loop
        for batch in range(bsz):
            t_batch = t[batch]

            x_t_class = x_t[batch, :]

            log_q_x_t_given_x_t_min_1 = self.log_Q_t_known_transitioning_to_class(t=t_batch, klass=x_t_class, device=device, cumulative=False)

            # TODO(will) - maybe just produce directly w/out log probabilities
            log_q_x_t_min_1_given_x_0 = self.log_Q_t(t=t_batch-1, device=device, cumulative=True)
            q_x_t_min_1_given_x_0 = log_q_x_t_min_1_given_x_0.exp()

            log_q_x_t_given_x_0 = self.log_Q_t_known_transitioning_to_class(t=t_batch, klass=x_t_class, device=device, cumulative=True)

            # TODO document here
            # - full matrices
            # - conversion between logspace and not

            # p(x_0) / q(x_t | x_0)
            log_step_1 = log_p_x_0[batch, :, :] - log_q_x_t_given_x_0
            step_1 = log_step_1.exp()

            # q(x_{t-1} | x_0=C_0) * p(x_0=C_0) / q(x_t | x_0=C_0) + ... + q(x_{t-1} | x_0=C_k) * p(x_0=C_k) / q(x_t | x_0=C_k)
            step_2 = q_x_t_min_1_given_x_0 @ step_1
            log_step_2 = step_2.log()

            # q(x_t | x_{t-1}) * [q(x_{t-1} | x_0=C_0) * p(x_0=C_0) / q(x_t | x_0=C_0) + ... + q(x_{t-1} | x_0=C_k) * p(x_0=C_k) / q(x_t | x_0=C_k)]
            log_step_3 = log_q_x_t_given_x_t_min_1 + log_step_2
            
            log_p_x_t_min_1_given_x_t[batch, :, :] = log_step_3 

        # TODO remove
        torch.save(log_p_x_t_min_1_given_x_t, f"/content/diffusers-out/log_p_x_t_min_1_given_x_t-{t_batch}.pt")

        return log_p_x_t_min_1_given_x_t

    def log_Q_t(self, t, device, cumulative: bool):
        """
        Returns the log probabilities of the transition matrix (cumulative or non-cumulative).
        
        See equation (7) for the non-cumulative transition matrix.

        Note that we don't have to return a separate transition matrix per latent pixel because 
        each latent pixel has the same transition matrix.

        Args:
            t (torch.Long):
                The timestep that determines which transition matrix is used.

            device (`torch.device`): 
                The device to place the returned matrix on.

            cumulative (`bool`):
                If cumulative is `False`, the columns are taken from the transition matrix `t-1`->`t`.
                If cumulative is `True`, the columns are taken from the transition matrix `0`->`t`.

        Returns:
            TODO add fixed dimensions for cumulative

            `torch.FloatTensor` of shape `(num classes, num classes)`:
                The log probabilities of the transition matrix (cumulative or non-cumulative).

                Where:
                - C_0 is a class of a latent pixel embedding
                - C_k is the class of the masked latent pixel

                non-cumulative result (omitting logarithms):
                q(x_t = C_0 | x_{t-1} = C_0) ... q(x_t = C_0 | x_{t-1} = C_k)
                               .      .                     .
                               .               .            .
                               .                      .     .
                q(x_t = C_k | x_{t-1} = C_0) ... q(x_t = C_k | x_{t-1} = C_k)

                cumulative result (omitting logarithms):
                q_cumulative(x_t = C_0 | x_0 = C_0) ... q_cumulative(x_t = C_0 | x_0 = C_k)
                                   .          .                        .
                                   .                   .               .
                                   .                          .        .
                q_cumulative(x_t = C_k | x_0 = C_0) ... q_cumulative(x_t = C_k | x_0 = C_k)

        """
        if cumulative:
            a = self.log_cumprod_at[t]
            b = self.log_cumprod_bt[t]
            c = self.log_cumprod_ct[t]

            shape = (self.num_embed, self.num_embed - 1)
        else:
            a = self.log_at[t]
            b = self.log_bt[t]
            c = self.log_ct[t]

            shape = (self.num_embed, self.num_embed)


        log_Q_t = torch.full(shape, b, device=device)
        log_Q_t.fill_diagonal_(a + b)
        log_Q_t[-1, :] = c

        if not cumulative:
            log_Q_t[:, -1] = self.min_logged_value
            log_Q_t[-1, -1] = 0 # 0 = log(1)

        return log_Q_t


    def log_Q_t_known_transitioning_to_class(self, *, t, klass, device, cumulative: bool):
        """
        Returns the log probabilities of the rows from the transition matrix 
        (cumulative or non-cumulative) for each latent pixel in `klass`.

        See equation (7) for the non-cumulative transition matrix.

        Args:
            t (torch.Long):
                The timestep that determines which transition matrix is used.

            klass (`torch.LongTensor` of shape `(num latent pixels)`):
                The classes of each latent pixel at time `t`.

            device (`torch.device`): 
                The device to place the returned matrix on.

            cumulative (`bool`):
                If cumulative is `False`, the columns are taken from the transition matrix `t-1`->`t`.
                If cumulative is `True`, the columns are taken from the transition matrix `0`->`t`.

        Returns:
            TODO add fixed dimensions for cumulative

            `torch.FloatTensor` of shape `(num classes, num latent pixels)`:
                Each _column_ of the returned matrix is a _row_ of log probabilities of the probability 
                transition matrix.
                
                Where:
                - `q_n` is the probability distribution for the forward process of the `n`th latent pixel.
                - C_0 is a class of a latent pixel embedding
                - C_k is the class of the masked latent pixel

                non-cumulative result (omitting logarithms):
                q_0(x_t | x_{t-1} = C_0) ... q_n(x_t | x_{t-1} = C_0)
                          .      .                     .
                          .               .            .
                          .                      .     .
                q_0(x_t | x_{t-1} = C_k) ... q_n(x_t | x_{t-1} = C_k)


                cumulative result (omitting logarithms):
                q_0_cumulative(x_t | x_0 = C_0) ... q_n_cumulative(x_t | x_0 = C_0)
                          .               .                          .
                          .                        .                 .
                          .                               .          .
                q_0_cumulative(x_t | x_0 = C_k) ... q_n_cumulative(x_t | x_0 = C_k)
        """
        num_latent_pixels = klass.shape[0]

        if cumulative:
            a = self.log_cumprod_at[t]
            b = self.log_cumprod_bt[t]
            c = self.log_cumprod_ct[t]

            shape = (self.num_embed - 1, num_latent_pixels)
        else:
            a = self.log_at[t]
            b = self.log_bt[t]
            c = self.log_ct[t]

            shape = (self.num_embed, num_latent_pixels)

        log_Q_t = torch.empty(shape, device=device)

        # Transitioning to masked latent pixels

        mask_class_mask = klass == self.mask_class

        log_Q_t[:, mask_class_mask] = c

        if not cumulative:
            log_Q_t[-1, mask_class_mask] = 0 # 0 == log(1)


        # Transitioning to non-masked latent pixels

        non_mask_class_mask = ~mask_class_mask

        log_Q_t[:, non_mask_class_mask] = b
        log_Q_t[klass[non_mask_class_mask], non_mask_class_mask] = a + b

        if not cumulative:
            log_Q_t[-1, non_mask_class_mask] = self.min_logged_value

        return log_Q_t
