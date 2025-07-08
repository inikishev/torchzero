from operator import itemgetter
from functools import partial
import math
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


def _lambertw_newton_raphson(x: TensorList, iterations=5):
    # z = torch.zeros_like(x)
    # mask_neg = x < 0
    # mask_pos = ~mask_neg

    # z[mask_pos] = torch.log(x[mask_pos] + 1.0)

    # x_neg = x[mask_neg]
    # z_neg = -1.0 + torch.sqrt(2.0 * (1.0 + math.e * x_neg))
    # z[mask_neg] = z_neg

    # x is always positive
    z = (x+1).log_()
    for _ in range(iterations):
        exp_z = z.exp()
        numerator = z * exp_z - x
        denominator = exp_z * (z + 1.0) + 1e-8
        delta = numerator / denominator
        z -= delta
    return z

# https://github.com/gmgeorg/torchlambertw/blob/main/torchlambertw/special.py
def _lambertw_winitzki(x: TensorList):
    x_log1p = x.log1p()
    return x_log1p * (1.0 - x_log1p.log1p() / (2.0 + x_log1p))


def adam_lambertw_(
    tensors: TensorList,
    exp_avg_: TensorList,
    exp_avg_xpx_: TensorList,
    alpha: float | NumberList,
    beta1: float | NumberList,
    beta2: float | NumberList,
    eps: float | NumberList,
    step: int,
    pow: float = 2,
    debiased: bool = True,
    max_exp_avg_xpx_: TensorList | None = None,
    iterations: int | None = 5,

    # inner args
    inner: Module | None = None,
    params: list[torch.Tensor] | None = None,
    grads: list[torch.Tensor] | None = None,
):
    """Returns new tensors."""
    tensors_abs = tensors.abs().clip_(max=20)
    tensors_xpx = tensors_abs.pow_(tensors_abs)
    exp_avg_xpx_.lerp_(tensors_xpx, 1-beta2)

    if max_exp_avg_xpx_ is not None:
        max_exp_avg_xpx_.maximum_(exp_avg_xpx_)
        exp_avg_xpx_ = max_exp_avg_xpx_

    if inner is not None:
        assert params is not None
        tensors = TensorList(apply_transform(inner, tensors, params=params, grads=grads))

    exp_avg_ = ema_(tensors, exp_avg_=exp_avg_, beta=beta1, dampening=0,lerp=True)
    if debiased: alpha = debiased_step_size(step, beta1=beta1, beta2=beta2, pow=pow, alpha=alpha)

    if iterations is None or iterations < 1: exp_avg_xpx_ = _lambertw_winitzki(exp_avg_xpx_)
    else: exp_avg_xpx_ = _lambertw_newton_raphson(exp_avg_xpx_, iterations)

    return (exp_avg_.lazy_mul(alpha) / exp_avg_xpx_.add_(eps))

class AdamLambertW(Transform):
    """Adam but uses abs x^x and LambertW instead of square and sqrt.
    The gradient will be clipped to 20 because float32 which you have to use otherwise you're PC will explode.

    Args:
        beta1 (float, optional): momentum. Defaults to 0.9.
        beta2 (float, optional): second momentum. Defaults to 0.999.
        eps (float, optional): epsilon. Defaults to 1e-8.
        alpha (float, optional): learning rate. Defaults to 1.
        amsgrad (bool, optional): Whether to divide by maximum of EMA of gradient squares instead. Defaults to False.
        pow (float, optional): power used in second momentum power and root. Defaults to 2.
        debiased (bool, optional): whether to apply debiasing to momentums based on current step. Defaults to True.
        iterations (int, optional): 0 or None means Winitzki approximation otherwise number of newton raphson iterations.
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
        iterations: int | None = 5,
        inner: Chainable | None = None
    ):
        defaults=dict(beta1=beta1,beta2=beta2,eps=eps,alpha=alpha,amsgrad=amsgrad,pow=pow,debiased=debiased, iterations=iterations)
        super().__init__(defaults, uses_grad=False)

        if inner is not None: self.set_child('inner', inner)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        beta1,beta2,eps,alpha=unpack_dicts(settings, 'beta1','beta2','eps','alpha', cls=NumberList)
        amsgrad,pow,debiased,iterations = itemgetter('amsgrad','pow','debiased','iterations')(settings[0])

        if amsgrad:
            exp_avg, exp_avg_xpx, max_exp_avg_xpx = unpack_states(states, tensors, 'exp_avg', 'exp_avg_xpx', 'max_exp_avg_xpx', cls=TensorList)
        else:
            exp_avg, exp_avg_xpx = unpack_states(states, tensors, 'exp_avg', 'exp_avg_xpx', cls=TensorList)
            max_exp_avg_xpx = None


        return adam_lambertw_(
            tensors=TensorList(tensors),
            exp_avg_=exp_avg,
            exp_avg_xpx_=exp_avg_xpx,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            step=step,
            pow=pow,
            debiased=debiased,
            max_exp_avg_xpx_=max_exp_avg_xpx,
            iterations=iterations,

            # inner args
            inner=self.children.get("inner", None),
            params=params,
            grads=grads,

        )
