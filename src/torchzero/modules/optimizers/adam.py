from collections import abc

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule

def _adam_step(ascent: TensorList, exp_avg: TensorList, exp_avg_sq: TensorList, lr, beta1, beta2, eps, step:int, max_exp_avg_sqs: TensorList | None):
    # Decay the first and second moment running average coefficient
    exp_avg.lerp_compat_(ascent, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(ascent, ascent.conj(), value=1 - beta2)

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    if max_exp_avg_sqs is not None:
        max_exp_avg_sqs.maximum_(exp_avg_sq)
        denom = (max_exp_avg_sqs.sqrt().div_(bias_correction2**0.5)).add_(eps)
    else:
        denom = (exp_avg_sq.sqrt().div_(bias_correction2**0.5)).add_(eps)
    return (exp_avg / denom).mul_(lr / bias_correction1)


class Adam(OptimizerModule):
    def __init__(self, lr: float = 1, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, amsgrad=False):
        """Adam. Combines momentum and RMSProp. Exactly matches pytorch adam.

        Args:
            lr (float, optional): learning rate. Defaults to 1.
            beta1 (float, optional): exponential decay rate of gradient moving average. Defaults to 0.9.
            beta2 (float, optional): exponential decay rate of squared gradient moving average. Defaults to 0.999.
            eps (float, optional): epsilon for numerical stability. Defaults to 1e-8.
            amsgrad (bool, optional): whether to use the AMSGrad variant of this algorithm from the paper
                On the Convergence of Adam and Beyond (default: False).
        """
        defaults = dict(lr = lr, beta1=beta1, beta2=beta2, eps=eps)
        super().__init__(defaults)

        self.cur_step = 1
        self.amsgrad = amsgrad

    @torch.no_grad
    def _update(self, state, ascent):
        settings = self.get_all_group_keys()

        if self.amsgrad:
            exp_avg, exp_avg_sq, max_exp_avg_sqs = self.get_state_keys(('exp_avg', 'exp_avg_sq', 'max_exp_avg_sqs'))
        else:
            exp_avg, exp_avg_sq = self.get_state_keys(('exp_avg', 'exp_avg_sq'))
            max_exp_avg_sqs = None

        updated_direction = _adam_step(
            ascent=ascent,
            exp_avg = exp_avg,
            exp_avg_sq = exp_avg_sq,
            lr = settings['lr'],
            beta1 = settings['beta1'],
            beta2 = settings['beta2'],
            eps = settings['eps'],
            step = self.cur_step,
            max_exp_avg_sqs = max_exp_avg_sqs,
        )
        self.cur_step += 1
        return updated_direction
