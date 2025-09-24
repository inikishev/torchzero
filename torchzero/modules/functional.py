"""
Arguments that are modified in-place are denoted with "_" at the end.

Some functions return one of the arguments which was modified in-place, some return new tensors.
Make sure to keep track of that to avoid unexpected in-place modifications of buffers. The returned
storage is always indicated in the docstring.

Additional functional variants are present in most module files, e.g. `adam_`, `rmsprop_`, `lion_`, etc.
"""
from collections.abc import Callable
from typing import overload

import torch

from ..utils import (
    NumberList,
    TensorList,
    generic_finfo_eps,
    generic_max,
    generic_sum,
    tofloat,
)

inf = float('inf')

def debiased_step_size(
    step,
    beta1: float | NumberList | None = None,
    beta2: float | NumberList | None = None,
    pow: float = 2,
    alpha: float | NumberList = 1,
):
    """returns multiplier to step size, step starts from 1"""
    if isinstance(beta1, NumberList): beta1 = beta1.fill_none(0)
    if isinstance(beta2, NumberList): beta2 = beta2.fill_none(0)

    step_size = alpha
    if beta1 is not None:
        bias_correction1 = 1.0 - (beta1 ** step)
        step_size /= bias_correction1
    if beta2 is not None:
        bias_correction2 = 1.0 - (beta2 ** step)
        step_size *= bias_correction2 ** (1/pow)
    return step_size

def debias(
    tensors_: TensorList,
    step: int,
    inplace: bool,
    beta1: float | NumberList | None = None,
    beta2: float | NumberList | None = None,
    alpha: float | NumberList = 1,
    pow: float = 2,
):
    step_size = debiased_step_size(step=step, beta1=beta1, beta2=beta2, pow=pow, alpha=alpha)
    if inplace: return tensors_.mul_(step_size)
    return tensors_ * step_size

def debias_second_momentum(tensors_:TensorList, step: int, beta: float | NumberList, pow: float, inplace:bool):
    """debias 2nd momentum, optionally in-place"""
    bias_correction2 = (1.0 - (beta ** step)) ** (1/pow)
    if inplace: return tensors_.div_(bias_correction2)
    return tensors_ / bias_correction2

def lerp_power_(tensors:TensorList, exp_avg_pow_:TensorList, beta:float|NumberList, pow:float) -> TensorList:
    """
    Lerp `exp_avg_pow_` with `tensors ^ pow`

    Returns `exp_avg_pow_`.
    """
    if pow == 1: return exp_avg_pow_.lerp_(tensors.abs(), 1-beta)
    if pow == 2: return exp_avg_pow_.mul_(beta).addcmul_(tensors, tensors, value = 1-beta)
    if pow % 2 == 0: return exp_avg_pow_.lerp_(tensors.pow(pow), 1-beta)
    return exp_avg_pow_.lerp_(tensors.pow(pow).abs_(), 1-beta)

def add_power_(tensors:TensorList, sum_:TensorList, pow:float) -> TensorList:
    """
    Add `tensors ^ pow` to `sum_`

    Returns `sum_`.
    """
    if pow == 1: return sum_.add_(tensors.abs())
    if pow == 2: return sum_.addcmul_(tensors, tensors)
    if pow % 2 == 0: return sum_.add_(tensors.pow(pow))
    return sum_.add_(tensors.pow(pow).abs_())


def root(tensors_:TensorList, p:float, inplace: bool):
    """
    Root of tensors, optionally in-place.

    Returns `tensors_` if `inplace` else new tensors.
    """
    if inplace:
        if p == 1: return tensors_.abs_()
        if p == 2: return tensors_.sqrt_()
        return tensors_.pow_(1/p)

    if p == 1: return tensors_.abs()
    if p == 2: return tensors_.sqrt()
    return tensors_.pow(1/p)


def ema_(
    tensors: TensorList,
    exp_avg_: TensorList,
    beta: float | NumberList,
    dampening: float | NumberList = 0,
    lerp: bool = True,
):
    """
    Updates `exp_avg_` with EMA of `tensors`.

    Returns `exp_avg_`.
    """
    tensors.lazy_mul_(1 - dampening)
    if lerp: return exp_avg_.lerp_(tensors, (1 - beta))
    return exp_avg_.mul_(beta).add_(tensors)

def ema_sq_(
    tensors: TensorList,
    exp_avg_sq_: TensorList,
    beta: float | NumberList,
    max_exp_avg_sq_: TensorList | None,
    pow: float = 2,
):
    """
    Updates `exp_avg_sq_` with EMA of squared `tensors`, if `max_exp_avg_sq_` is not None, updates it with maximum of EMA.

    Returns `exp_avg_sq_` or `max_exp_avg_sq_`.
    """
    lerp_power_(tensors=tensors, exp_avg_pow_=exp_avg_sq_,beta=beta,pow=pow)

    # AMSGrad
    if max_exp_avg_sq_ is not None:
        max_exp_avg_sq_.maximum_(exp_avg_sq_)
        exp_avg_sq_ = max_exp_avg_sq_

    return exp_avg_sq_

def sqrt_ema_sq_(
    tensors: TensorList,
    exp_avg_sq_: TensorList,
    beta: float | NumberList,
    max_exp_avg_sq_: TensorList | None,
    debiased: bool,
    step: int,
    pow: float = 2,
    ema_sq_fn: Callable = ema_sq_,
):
    """
    Updates `exp_avg_sq_` with EMA of squared `tensors` and calculates it's square root,
    with optional AMSGrad and debiasing.

    Returns new tensors.
    """
    exp_avg_sq_=ema_sq_fn(
        tensors=tensors,
        exp_avg_sq_=exp_avg_sq_,
        beta=beta,
        max_exp_avg_sq_=max_exp_avg_sq_,
        pow=pow,
    )

    sqrt_exp_avg_sq = root(exp_avg_sq_, pow, inplace=False)

    if debiased: sqrt_exp_avg_sq = debias_second_momentum(sqrt_exp_avg_sq, step=step, beta=beta, pow=pow, inplace=True)
    return sqrt_exp_avg_sq


def centered_ema_sq_(tensors: TensorList, exp_avg_: TensorList, exp_avg_sq_: TensorList,
                     beta: float | NumberList, max_exp_avg_sq_: TensorList | None = None, pow:float=2):
    """
    Updates `exp_avg_` and `exp_avg_sq_` with EMA of `tensors` and squared `tensors`,
    centers `exp_avg_sq_` by subtracting `exp_avg_` squared.

    Returns `max_exp_avg_sq_` or new tensors.
    """
    exp_avg_sq_ = ema_sq_(tensors, exp_avg_sq_=exp_avg_sq_, beta=beta, max_exp_avg_sq_=max_exp_avg_sq_, pow=pow)
    exp_avg_.lerp_(tensors, 1-beta)
    exp_avg_sq_ = exp_avg_sq_.addcmul(exp_avg_, exp_avg_, value=-1)

    # AMSGrad
    if max_exp_avg_sq_ is not None:
        max_exp_avg_sq_.maximum_(exp_avg_sq_)
        exp_avg_sq_ = max_exp_avg_sq_

    return exp_avg_sq_

def sqrt_centered_ema_sq_(
    tensors: TensorList,
    exp_avg_: TensorList,
    exp_avg_sq_: TensorList,
    max_exp_avg_sq_: TensorList | None,
    beta: float | NumberList,
    debiased: bool,
    step: int,
    pow: float = 2,
):
    """
    Updates `exp_avg_` and `exp_avg_sq_` with EMA of `tensors` and squared `tensors`,
    centers `exp_avg_sq_` by subtracting `exp_avg_` squared. Calculates it's square root,
    with optional AMSGrad and debiasing.

    Returns new tensors.
    """
    return sqrt_ema_sq_(
        tensors=tensors,
        exp_avg_sq_=exp_avg_sq_,
        beta=beta,
        max_exp_avg_sq_=max_exp_avg_sq_,
        debiased=debiased,
        step=step,
        pow=pow,
        ema_sq_fn=lambda *a, **kw: centered_ema_sq_(*a, **kw, exp_avg_=exp_avg_)
    )

def initial_step_size(tensors: torch.Tensor | TensorList, eps=None) -> float:
    """initial scaling taken from pytorch L-BFGS to avoid requiring a lot of line search iterations,
    this version is safer and makes sure largest value isn't smaller than epsilon."""
    tensors_abs = tensors.abs()
    tensors_sum = generic_sum(tensors_abs)
    tensors_max = generic_max(tensors_abs)

    feps = generic_finfo_eps(tensors)
    if eps is None: eps = feps
    else: eps = max(eps, feps)

    # scale should not make largest value smaller than epsilon
    min = eps / tensors_max
    if min >= 1: return 1.0

    scale = 1 / tensors_sum
    scale = scale.clip(min=min.item(), max=1)
    return scale.item()


def epsilon_step_size(tensors: torch.Tensor | TensorList, alpha=1e-7) -> float:
    """makes sure largest value isn't smaller than epsilon."""
    tensors_abs = tensors.abs()
    tensors_max = generic_max(tensors_abs)
    if tensors_max < alpha: return 1.0

    if tensors_max < 1: alpha = alpha / tensors_max
    return tofloat(alpha)



def safe_clip(x: torch.Tensor, min=None):
    """makes sure absolute value of scalar tensor x is not smaller than min"""
    assert x.numel() == 1, x.shape
    if min is None: min = torch.finfo(x.dtype).tiny * 2

    if x.abs() < min: return x.new_full(x.size(), min).copysign(x)
    return x


def clip_by_finfo(x, finfo: torch.finfo):
    """clips by (dtype.max / 2, dtype.min / 2)"""
    if x > finfo.max / 2: return finfo.max / 2
    if x < finfo.min / 2: return finfo.min / 2
    return x