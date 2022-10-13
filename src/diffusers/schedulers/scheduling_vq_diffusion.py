from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract_schedule_parameters(schedule_parameters, t):
    bsz = t.shape[0]
    out = schedule_parameters.gather(-1, t)
    return out.reshape(bsz, 1, 1)


# Takes an input of (potentially batched) index vectors.
# Returns the log of the one hots as column vectors.
def index_to_log_onehot(x, num_classes):
    x_onehot = F.one_hot(x, num_classes)

    # TODO(will) - this permute order isn't clear, can we just always
    # assume the input is 3dim or if the output is 2dim, can we just
    # do a check?
    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def log_sample_categorical(logits, num_classes):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample = (gumbel_noise + logits).argmax(dim=1)
    log_sample = index_to_log_onehot(sample, num_classes)
    return log_sample

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

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
    ...


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
        num_classes: int,
        num_train_timesteps: int = 100,
        min_logged_value: float = -70.0,
    ):
        self.num_classes = num_classes
        self.min_logged_value = min_logged_value

        # By convention, the index for the mask class is the last class index
        self.mask_class = self.num_classes - 1

        at, att = alpha_schedules(num_train_timesteps)
        ct, ctt = gamma_schedules(num_train_timesteps)

        num_non_mask_classes = self.num_classes - 1
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

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.log_at = log_at.float()
        self.log_bt = log_bt.float()
        self.log_ct = log_ct.float()
        self.log_cumprod_at = log_cumprod_at.float()
        self.log_cumprod_bt = log_cumprod_bt.float()
        self.log_cumprod_ct = log_cumprod_ct.float()
        self.log_1_min_ct = log_1_min_ct.float()
        self.log_1_min_cumprod_ct = log_1_min_cumprod_ct.float()

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
        timesteps = torch.from_numpy(np.arange(0, self.num_inference_steps)[::-1].copy())
        self.timesteps = torch.from_numpy(timesteps).to(device)


    def step(self, *, log_x_0, log_x_t, t) -> Union[VQDiffusionSchedulerOutput, Tuple]:
        """
        log_x_start is the log probs of the unnoised image probabilities. probabilities as col vectors
        (batch, )

        i.e. log_x_start[batch, class probability, latent pixel]
        """
        log_model_pred = self.q_posterior(log_x_0=log_x_0, log_x_t=log_x_t, t=t)

        out = log_sample_categorical(log_model_pred)

        return out

    def q_posterior(self, *, log_x_0, log_x_t, t):
        """
        Calculates Equation 11 in log space

        p(x_{t-1} | x_t, y) = sum( q(x_{t-1} | x_t, x_0') p(x_0' | x_t, y) )

        Where: 
        - The sum is over the predicted classes for the denoised image.
        - Writing \tilde{x}_{0} as x_0'.
        - Writing p_{\theta} as p
        - x_0' is the noiseless token distribution predicted by the transformer. 

        Args:
            TODO

        Returns:
            TODO
        """
        bsz = log_x_0.size()[0]

        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.mask_class).unsqueeze(1)

        # TODO probably have to add device
        log_one_vector = torch.zeros(bsz, 1, 1)

        # equivalent to `torch.zeros((bsz, 1, self.inner_dim)).log().clamp(self.min_logged_value)`
        # NOTE will be slightly different than prev min value
        log_zero_vector = torch.fill((bsz, 1, self.inner_dim), self.min_logged_value)

        log_qt = self.q_pred(log_x_t, t)  # q(xt|x0)
        log_qt = log_qt[:, :-1, :]
        log_cumprod_ct = extract_schedule_parameters(self.log_cumprod_ct, t)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.mask_class, -1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1)
        log_ct = extract_schedule_parameters(self.log_ct, t)  # ct
        ct_vector = log_ct.expand(-1, self.mask_class, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        q = log_x_0[:, :-1, :] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)


    def q_pred(self, log_x_0, t):
        """
        Calculates equation 4 in log space over a batch of class predictions

        q(x_t | x_0) = v^T(x_t) Q_cumprod_t v(x_0)
                     = Q_cumprod_t[x_t][x_0]

        v(x) is the one-hot column vector with the one-hot value at index x.

        Q_cumprod_t[m][n] = q(x_t = m | x_0 = n)

        Args:
            log_x_0 (`torch.FloatTensor` of shape `(batch_size, log probability of class, latent pixel)`): 
                The log probabilities of the latent pixel classes at time step 0

            t (`torch.LongTensor` of shape `(batch_size,)`):
                Current diffusion steps
        Returns:
            `torch.FloatTensor` of shape `(batch_size, log probability of class, latent pixel)`:
                The log probabilities of the latent pixel classes at time step `t`. Including the
                0 log probs that the initial image is masked.
        """
        log_cumprod_at = extract_schedule_parameters(self.log_cumprod_at, t)
        log_cumprod_bt = extract_schedule_parameters(self.log_cumprod_bt, t)
        log_cumprod_ct = extract_schedule_parameters(self.log_cumprod_ct, t)
        log_1_min_cumprod_ct = extract_schedule_parameters(self.log_1_min_cumprod_ct, t)

        # See `q_pred_one_timestep`` for docs on log_probs_unmasked_classes and log_probe_masked_class.
        # The same explanation holds for the 0 -> t
        log_probs_unmasked_classes = log_add_exp(log_cumprod_at + log_x_0[:, :-1, :], log_cumprod_bt)
        log_probs_masked_class = log_add_exp(log_1_min_cumprod_ct + log_x_0[:, -1:, :], log_cumprod_ct)

        log_x_t = torch.cat((log_probs_unmasked_classes, log_probs_masked_class), dim=1)

        return log_x_t

    def q_pred_one_timestep(self, log_x_t, t):
        """
        Calculates equation 3 in log space over a batch of class predictions

        q(x_t | x_{t-1}) = v^T(x_t) Q_t v(x_{t-1})
                         = Q_t[x_t][x_{t-1}]

        v(x) is the one-hot column vector with the one-hot value at index x.

        Q_t[m][n] = q(x_t = m | x_{t-1} = n)

        Args:
            log_x_t (`torch.FloatTensor` of shape `(batch_size, log probability of class, latent pixel)`): 
                The log probabilities of the latent pixel classes at time steps `t`

            t (`torch.LongTensor` of shape `(batch_size,)`):
                Current diffusion steps
        Returns:
            `torch.FloatTensor` of shape `(batch_size, log probability of class, latent pixel)`:
                The log probabilities of the latent pixel classes at time step `t + 1`
        """
        log_at = extract_schedule_parameters(self.log_at, t)
        log_bt = extract_schedule_parameters(self.log_bt, t)
        log_ct = extract_schedule_parameters(self.log_ct, t)
        log_1_min_ct = extract_schedule_parameters(self.log_1_min_ct, t)

        # q(x_{t+1} = a non masked class) = a_t * p(x_t = the same non masked class) + b_t
        #                                  |_______________________________________|  |___|
        #                                                       |                       |
        #                                                       |                       uniform resampling to the same class
        #                                                       |
        #                                                       remain same class
        log_probs_unmasked_classes = log_add_exp(log_at + log_x_t[:, :-1, :], log_bt)

        # q(x_{t+1} = masked) = (1 - c_t) * p(x_t = masked) + c_t
        #                     = p(x_t = masked) - c_t * p(x_t = masked) + c_t
        #                     = p(x_t = masked) + c_t * (1 - p(x_t = masked))
        #                      |______________|  |___________________________|
        #                              |                      |
        #                              |                      x_t was not masked and was masked on t -> t+1
        #                              |
        #                              x_t was already masked on some prior step
        log_probs_masked_class = log_add_exp(log_1_min_ct + log_x_t[:, -1:, :], log_ct)

        log_x_t_plus_1 = torch.cat((log_probs_unmasked_classes, log_probs_masked_class), dim=1)

        return log_x_t_plus_1
