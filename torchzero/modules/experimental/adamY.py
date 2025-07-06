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


def adamy_(
    p: TensorList,
    p_prev: TensorList,
    g: TensorList,
    g_prev: TensorList,
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
    if step == 1:
        p_prev.copy_(p)
        g_prev.copy_(g)

        update = g.clip(-0.1,0.1).lazy_mul_(alpha)
        if params_ is None: return update
        params_.sub_(update)
        return None

    s = p-p_prev
    y = (g-g_prev).div_(s.global_vector_norm().clip(min=1e-8))
    p_prev.copy_(p)
    g_prev.copy_(g)

    exp_avg_ = ema_(g, exp_avg_=exp_avg_, beta=beta1, dampening=0,lerp=True)

    sqrt_exp_avg_sq = sqrt_ema_sq_(y, exp_avg_sq_=exp_avg_sq_, beta=beta2, max_exp_avg_sq_=max_exp_avg_sq_,
                                   debiased=False,step=step,pow=pow)

    if debiased: alpha = debiased_step_size(step, beta1=beta1, beta2=beta2, pow=pow, alpha=alpha)

    # params is None, return update
    if params_ is None: return (exp_avg_ / sqrt_exp_avg_sq.add_(eps)).lazy_mul(alpha)

    # update params in-place
    params_.addcdiv_(exp_avg_, sqrt_exp_avg_sq.add_(eps), -alpha)
    return None

class AdamY(Module):
    """Adam but uses scaled gradient differences for second momentum.

    Verdict: I haven't tested this yet.

    .. warning::
        Experimental.
    """
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
        defaults=dict(beta1=beta1,beta2=beta2,eps=eps,alpha=alpha,amsgrad=amsgrad,pow=pow,debiased=debiased)
        super().__init__(defaults)
        self.getter = itemgetter('amsgrad','pow','debiased')

    @torch.no_grad
    def step(self, var):
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        beta1,beta2,eps,alpha=self.get_settings(var.params, 'beta1','beta2','eps','alpha', cls=NumberList)
        amsgrad,pow,debiased = self.getter(self.settings[var.params[0]])

        if amsgrad:
            exp_avg, exp_avg_sq, max_exp_avg_sq = self.get_state(var.params,'exp_avg','exp_avg_sq','max_exp_avg_sq', cls=TensorList)
        else:
            exp_avg, exp_avg_sq = self.get_state(var.params, 'exp_avg','exp_avg_sq', cls=TensorList)
            max_exp_avg_sq = None

        # if this is last module, update parameters in-place with slightly more efficient addcdiv_
        if var.is_last:
            if var.last_module_lrs is not None: alpha = alpha * var.last_module_lrs
            passed_params = TensorList(var.params)
            var.stop = True
            var.skip_update = True

        else:
            passed_params = None

        p_prev = self.get_state(var.params, 'p_prev', cls=TensorList)
        g_prev = self.get_state(var.params, 'g_prev', cls=TensorList)


        var.update = adamy_(
            p=TensorList(var.params),
            p_prev=p_prev,
            g=TensorList(var.get_update()),
            g_prev=g_prev,
            exp_avg_=exp_avg,
            exp_avg_sq_=exp_avg_sq,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            step=step,
            pow=pow,
            debiased=debiased,
            max_exp_avg_sq_=max_exp_avg_sq,
            params_=passed_params,
        )

        return var
