from dataclasses import dataclass
from typing import Tuple, Union

import torch
import numpy as np
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin

def index_to_log_onehot(x, num_classes):
    x_onehot = F.one_hot(x, num_classes)
    x_onehot = x_onehot.permute(0, 2, 1)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

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

        # TODO remove the `to('cuda')`s and have these switch devices along with the pipeline
        self.log_at = log_at.float().to('cuda')
        self.log_bt = log_bt.float().to('cuda')
        self.log_ct = log_ct.float().to('cuda')
        self.log_cumprod_at = log_cumprod_at.float().to('cuda')
        self.log_cumprod_bt = log_cumprod_bt.float().to('cuda')
        self.log_cumprod_ct = log_cumprod_ct.float().to('cuda')

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
        log_p_x_t_min_1 = self.q_posterior(log_p_x_0, x_t, t)

        log_p_x_t_min_1 = gumbel_noised(log_p_x_t_min_1)

        x_t_min_1 = log_p_x_t_min_1.argmax(dim=1)

        if not return_dict:
            return (x_t_min_1,)

        return VQDiffusionSchedulerOutput(x_t_min_1=x_t_min_1)

    def q_posterior(self, log_p_x_0, x_t, t):
        class_log_onehot = index_to_log_onehot(x_t, self.num_embed)

        log_q_x_t_given_x_0 = self.log_Q_t_transitioning_to_known_class(t=t, klass=x_t, class_log_onehot=class_log_onehot, cumulative=True)

        log_q_t_given_x_t_min_1 = self.log_Q_t_transitioning_to_known_class(t=t, klass=x_t, class_log_onehot=class_log_onehot, cumulative=False)
        
        # p_0(x_0=C_0 | x_t) / q(x_t | x_0=C_0)          ...      p_n(x_0=C_0 | x_t) / q(x_t | x_0=C_0)
        #               .                    .                                   .
        #               .                            .                           .
        #               .                                      .                 .
        # p_0(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1})  ...      p_n(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1})
        q = log_p_x_0 - log_q_x_t_given_x_0

        # p_0(x_0=C_0 | x_t) / q(x_t | x_0=C_0) + ... + p_0(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1}), ... ,
        # p_n(x_0=C_0 | x_t) / q(x_t | x_0=C_0) + ... + p_n(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1})
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)

        q = q - q_log_sum_exp

        q = self.xqpred(q, t - 1)

        log_p_x_t_min_1 = q + log_q_t_given_x_t_min_1 + q_log_sum_exp

        return log_p_x_t_min_1

    # TODO what to name this?
    def xqpred(self, q, t):
        a = self.log_cumprod_at[t]
        b = self.log_cumprod_bt[t]
        c = self.log_cumprod_ct[t]

        num_latent_pixels = q.shape[2]
        c = c.expand(1, 1, num_latent_pixels)

        q = (q + a).logaddexp(b)
        q = torch.cat((q, c), dim=1)

        return q


    def log_Q_t_transitioning_to_known_class(self, *, t: torch.int, klass: torch.LongTensor, class_log_onehot: torch.FloatTensor, cumulative: bool):
        """
        Returns the log probabilities of the rows from the (cumulative or non-cumulative) transition matrix 
        for each latent pixel in `klass`.

        See equation (7) for the complete non-cumulative transition matrix. The complete cumulative transition
        matrix is the same structure except the parameters (alpha, beta, gamma) are the cumulative analogs.

        Args:
            t (torch.Long):
                The timestep that determines which transition matrix is used.

            klass (`torch.LongTensor` of shape `(batch size, num latent pixels)`):
                The classes of each latent pixel at time `t`.

            class_log_onehot (`torch.FloatTensor` of shape `(batch size, num classes, num latent pixels)`):
                The log one-hot vectors of `klass`

            cumulative (`bool`):
                If cumulative is `False`, the columns are taken from the transition matrix `t-1`->`t`.
                If cumulative is `True`, the columns are taken from the transition matrix `0`->`t`.

        Returns:
            `torch.FloatTensor` of shape `(batch size, num classes - 1, num latent pixels)`:
                Each _row_ of the returned matrix is a _row_ of log probabilities of the complete
                probability transition matrix.

                When non cumulative, returns `self.num_classes - 1` rows because the initial latent 
                pixel cannot be masked
                
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
                q_0_cumulative(x_t | x_0 = C_{k-1}) ... q_n_cumulative(x_t | x_0 = C_{k-1})
        """
        if cumulative:
            a = self.log_cumprod_at[t]
            b = self.log_cumprod_bt[t]
            c = self.log_cumprod_ct[t]
        else:
            a = self.log_at[t]
            b = self.log_bt[t]
            c = self.log_ct[t]

        if not cumulative:
            # The values in the onehot vector can also be used as the logprobs for transitioning
            # from masked latent pixels. If we are not calculating the cumulative transitions,
            # we need to save these vectors to be re-appended to the final matrix so the values
            # aren't overwritten.
            #
            # `P(x_t!=mask|x_{t-1=mask}) = 0` and 0 will be the value of the last row of the onehot vector
            # if x_t is not masked
            #
            # `P(x_t=mask|x_{t-1=mask}) = 1` and 1 will be the value of the last row of the onehot vector
            # if x_t is masked
            class_log_onehot_transitioning_from_masked = class_log_onehot[:, -1, :].unsqueeze(0)

        # `index_to_log_onehot` will add onehot vectors for masked pixels,
        # so the default one hot matrix has one too many rows. See the doc string
        # for an explanation of the dimensionality of the returned matrix.
        class_log_onehot = class_log_onehot[:, :-1, :]

        # this is a cheeky trick to produce the transition probabilities using log one-hot vectors.
        #
        # Don't worry about what values this sets in the columns that mark transitions
        # to masked latent pixels. They are overwrote later with the `mask_class_mask`.
        #
        # Looking at the below logspace formula in base 10, each value will evaluate to either 
        # `1 * a + b = a + b` where `log_Q_t` has the one hot value in the column
        # or 
        # `0 * a + b = b` where `log_Q_t` has the 0 values in the column.
        #
        # See equation 7 for more details.
        log_Q_t = (class_log_onehot + a).logaddexp(b)

        # The whole column of each masked pixel is `c`
        mask_class_mask = klass == self.mask_class
        mask_class_mask = mask_class_mask.unsqueeze(1).expand(-1, self.num_embed - 1, -1)
        log_Q_t[mask_class_mask] = c

        if not cumulative:
            log_Q_t = torch.cat((log_Q_t, class_log_onehot_transitioning_from_masked), dim=1)

        return log_Q_t
