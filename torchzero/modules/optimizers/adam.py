from operator import itemgetter
from functools import partial

import torch

from ...core import Module, Target, Transform
from ...utils import NumberList, TensorList
from ..functional import (
    debias1,
    ema_,
    sqrt_ema_sq_,
)
from ..lr.lr import lazy_lr
from ..momentum.experimental import sqrt_nag_ema_sq_
from ..momentum.momentum import nag_


def adam_(
    tensors: TensorList,
    exp_avg_: TensorList,
    exp_avg_sq_: TensorList,
    alpha: float | NumberList,
    beta1: float | NumberList,
    beta2: float | NumberList,
    eps: float | NumberList,
    step: int,
    pow: float = 2,
    debiased: bool = True,
    max_exp_avg_sq_: TensorList | None = None,
    params_: TensorList | None = None,
):
    """Returns new tensors or updates params in-place."""
    exp_avg_ = ema_(tensors, exp_avg_=exp_avg_, beta=beta1, dampening=0,lerp=True)

    if debiased: exp_avg_ = debias1(exp_avg_, step=step, beta=beta1, alpha=alpha, inplace=False)
    else: exp_avg_ = lazy_lr(exp_avg_, lr=alpha, inplace=False)

    sqrt_exp_avg_sq = sqrt_ema_sq_(tensors, exp_avg_sq_=exp_avg_sq_, beta=beta2, max_exp_avg_sq_=max_exp_avg_sq_,
                                   debiased=debiased,step=step,pow=pow).add_(eps)
    # params is None, return update
    if params_ is None: return exp_avg_ / sqrt_exp_avg_sq

    # update params in-place
    params_.addcdiv_(exp_avg_, sqrt_exp_avg_sq, -1)

class Adam(Module):
    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad: bool = False,
        alpha: float = 1.,
        pow: float = 2,
        debiased: bool = True,
    ):
        """Adam. This matches pytorch Adam implementation.

        Args:
            beta1 (float, optional): momentum. Defaults to 0.9.
            beta2 (float, optional): second momentum (empirical fisher diagonal). Defaults to 0.999.
            eps (float, optional): epsilon. Defaults to 1e-8.
            alpha (float, optional): learning rate. Defaults to 1.
            amsgrad (bool, optional): Whether to use AMSGrad. Defaults to False.
            pow (float, optional): power used in second momentum power and root. Defaults to 2.
            debiased (bool, optional): whether to apply debiasing to momentums based on current step. Defaults to True.
        """
        defaults=dict(beta1=beta1,beta2=beta2,eps=eps,alpha=alpha,amsgrad=amsgrad,pow=pow,debiased=debiased)
        super().__init__(defaults)
        self.current_step = 0
        self.getter = itemgetter('amsgrad','pow','debiased')

    @torch.no_grad
    def step(self, vars):
        self.current_step += 1

        beta1,beta2,eps,alpha=self.get_settings('beta1','beta2','eps','alpha', params=vars, cls=NumberList)
        amsgrad,pow,debiased = self.getter(self.defaults)

        if amsgrad:
            exp_avg, exp_avg_sq, max_exp_avg_sq = self.get_state('exp_avg','exp_avg_sq','max_exp_avg_sq', params=vars, cls=TensorList)
        else:
            exp_avg, exp_avg_sq = self.get_state('exp_avg','exp_avg_sq', params=vars, cls=TensorList)
            max_exp_avg_sq = None

        # if this is last module, update parameters in-place with slightly more efficient addcdiv_
        if vars.is_last:
            if vars.last_module_lrs is not None: alpha = alpha * vars.last_module_lrs
            passed_params = TensorList(vars.params)
            vars.stop = True

        else:
            passed_params = None

        vars.update = adam_(
            tensors=TensorList(vars.get_update()),
            exp_avg_=exp_avg,
            exp_avg_sq_=exp_avg_sq,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            step=self.current_step,
            pow=pow,
            debiased=debiased,
            max_exp_avg_sq_=max_exp_avg_sq,
            params_=passed_params,
        )

        return vars
