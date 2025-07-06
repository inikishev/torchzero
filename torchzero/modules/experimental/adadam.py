from operator import itemgetter
from functools import partial

import torch

from ...core import Module, Target, Transform
from ...utils import NumberList, TensorList
from ..functional import (
    debias, debiased_step_size,
    ema_,
    sqrt_ema_sq_,
)
from ..step_size.lr import lazy_lr
from ..momentum.experimental import sqrt_nag_ema_sq_
from ..momentum.momentum import nag_


def adadam_(
    tensors: TensorList,
    exp_avg_: TensorList,
    exp_avg_sq_: TensorList,
    exp_avg_qu_: TensorList,
    alpha: float | NumberList,
    beta1: float | NumberList,
    beta2: float | NumberList,
    precond_beta: float | NumberList,
    eps: float | NumberList,
    step: int,
    pow: float = 2,
    debiased: bool = True,
    max_exp_avg_sq_: TensorList | None = None,
    max_exp_avg_qu_: TensorList | None = None,
    params_: TensorList | None = None,
):
    """Returns new tensors or updates params in-place."""
    exp_avg_ = ema_(tensors, exp_avg_=exp_avg_, beta=beta1, dampening=0,lerp=True)

    sqrt_exp_avg_sq = sqrt_ema_sq_(tensors, exp_avg_sq_=exp_avg_sq_, beta=beta2, max_exp_avg_sq_=max_exp_avg_sq_,
                                   debiased=False,step=step,pow=pow)
    sqrt_exp_avg_qu = sqrt_ema_sq_(tensors/(sqrt_exp_avg_sq+1e-8), exp_avg_sq_=exp_avg_qu_,
                                   beta=precond_beta,max_exp_avg_sq_=max_exp_avg_qu_, debiased=False,step=step,pow=pow)

    if debiased: alpha = debiased_step_size(step, beta1=beta1, beta2=beta2, pow=pow, alpha=alpha)

    # params is None, return update
    if params_ is None: return (exp_avg_ / sqrt_exp_avg_qu.add_(eps)).lazy_mul(alpha)

    # update params in-place
    params_.addcdiv_(exp_avg_, sqrt_exp_avg_qu.add_(eps), -alpha)
    return None

class Adadam(Module):
    """Adam with a diagonally preconditioned preconditioner.

    Verdict: I haven't tested this yet.

    .. warning::
        Experimental.
    """
    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.999,
        precond_beta: float = 0.999,
        eps: float = 1e-8,
        amsgrad: bool = False,
        alpha: float = 1.,
        pow: float = 2,
        debiased: bool = True,
    ):
        defaults=dict(beta1=beta1,beta2=beta2,precond_beta=precond_beta,eps=eps,alpha=alpha,amsgrad=amsgrad,pow=pow,debiased=debiased)
        super().__init__(defaults)
        self.getter = itemgetter('amsgrad','pow','debiased')

    @torch.no_grad
    def step(self, var):
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1
        params = var.params

        beta1,beta2,precond_beta,eps,alpha=self.get_settings(params, 'beta1','beta2','precond_beta','eps','alpha', cls=NumberList)
        amsgrad,pow,debiased = self.getter(self.settings[var.params[0]])

        if amsgrad:
            exp_avg, exp_avg_sq, exp_avg_qu, max_exp_avg_sq, max_exp_avg_qu = self.get_state(params, 'exp_avg','exp_avg_sq', 'exp_avg_qu', 'max_exp_avg_sq', 'max_exp_avg_qu', cls=TensorList)
        else:
            exp_avg, exp_avg_sq, exp_avg_qu = self.get_state(params, 'exp_avg','exp_avg_sq', 'exp_avg_qu', cls=TensorList)
            max_exp_avg_sq = None
            max_exp_avg_qu = None

        # if this is last module, update parameters in-place with slightly more efficient addcdiv_
        if var.is_last:
            if var.last_module_lrs is not None: alpha = alpha * var.last_module_lrs
            passed_params = TensorList(var.params)
            var.stop = True
            var.skip_update = True

        else:
            passed_params = None

        var.update = adadam_(
            tensors=TensorList(var.get_update()),
            exp_avg_=exp_avg,
            exp_avg_sq_=exp_avg_sq,
            exp_avg_qu_=exp_avg_qu,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            precond_beta=precond_beta,
            eps=eps,
            step=step,
            pow=pow,
            debiased=debiased,
            max_exp_avg_sq_=max_exp_avg_sq,
            max_exp_avg_qu_=max_exp_avg_qu,
            params_=passed_params,
        )

        return var
