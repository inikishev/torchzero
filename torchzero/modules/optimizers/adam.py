from operator import itemgetter
from functools import partial

import torch

from ...core import Module, Target, Transform, apply_transform, Chainable
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states
from ..functional import (
    debias, debiased_step_size,
    ema_,
    sqrt_ema_sq_,
)
from ..step_size.lr import lazy_lr
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

    # inner args
    inner: Module | None = None,
    params: list[torch.Tensor] | None = None,
    grads: list[torch.Tensor] | None = None,
):
    """Returns new tensors."""
    sqrt_exp_avg_sq = sqrt_ema_sq_(tensors, exp_avg_sq_=exp_avg_sq_, beta=beta2, max_exp_avg_sq_=max_exp_avg_sq_,
                                   debiased=False,step=step,pow=pow)

    if inner is not None:
        assert params is not None
        tensors = TensorList(apply_transform(inner, tensors, params=params, grads=grads))

    exp_avg_ = ema_(tensors, exp_avg_=exp_avg_, beta=beta1, dampening=0,lerp=True)
    if debiased: alpha = debiased_step_size(step, beta1=beta1, beta2=beta2, pow=pow, alpha=alpha)
    return (exp_avg_.lazy_mul(alpha) / sqrt_exp_avg_sq.add_(eps))

class Adam(Transform):
    """Adam. Divides gradient EMA by EMA of gradient squares with debiased step size.

    This implementation is identical to :code:`torch.optim.Adam`.

    Args:
        beta1 (float, optional): momentum. Defaults to 0.9.
        beta2 (float, optional): second momentum. Defaults to 0.999.
        eps (float, optional): epsilon. Defaults to 1e-8.
        alpha (float, optional): learning rate. Defaults to 1.
        amsgrad (bool, optional): Whether to divide by maximum of EMA of gradient squares instead. Defaults to False.
        pow (float, optional): power used in second momentum power and root. Defaults to 2.
        debiased (bool, optional): whether to apply debiasing to momentums based on current step. Defaults to True.
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
        inner: Chainable | None = None
    ):
        defaults=dict(beta1=beta1,beta2=beta2,eps=eps,alpha=alpha,amsgrad=amsgrad,pow=pow,debiased=debiased)
        super().__init__(defaults, uses_grad=False)

        if inner is not None: self.set_child('inner', inner)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        beta1,beta2,eps,alpha=unpack_dicts(settings, 'beta1','beta2','eps','alpha', cls=NumberList)
        amsgrad,pow,debiased = itemgetter('amsgrad','pow','debiased')(settings[0])

        if amsgrad:
            exp_avg, exp_avg_sq, max_exp_avg_sq = unpack_states(states, tensors, 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq', cls=TensorList)
        else:
            exp_avg, exp_avg_sq = unpack_states(states, tensors, 'exp_avg', 'exp_avg_sq', cls=TensorList)
            max_exp_avg_sq = None


        return adam_(
            tensors=TensorList(tensors),
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

            # inner args
            inner=self.children.get("inner", None),
            params=params,
            grads=grads,

        )
