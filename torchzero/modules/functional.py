"""
Arguments that are modified in-place are denoted with "_" at the end.

Some functions return one of the arguments which was modified in-place, some return new tensors.
Make sure to keep track of that to avoid unexpected in-place modifications of buffers. The returned
storage is always indicated in the docstring.

Additional functional variants are present in most module files, e.g. `adam_`, `rmsprop_`, `lion_`, etc.
"""

from collections.abc import Callable, Sequence

from ..utils import NumberList, TensorList

inf = float('inf')
def bias_correction1_step_size(step, beta: float | NumberList, alpha: float | NumberList):
    """returns step size"""
    return alpha / (1 - beta**step)

def bias_correction2_value(step, beta: float | NumberList, pow: float):
    """square root of second momentum should be divided by this"""
    return (1 - beta**step) ** (1 / pow)

def debias1(tensors_:TensorList, step: int, beta: float | NumberList, alpha:float | NumberList, inplace:bool):
    """debias 1st momentum, optionally in-place"""
    if inplace: return tensors_.mul_(bias_correction1_step_size(step, beta=beta, alpha=alpha))
    return tensors_ * bias_correction1_step_size(step, beta=beta, alpha=alpha)

def debias2(tensors_:TensorList, step: int, beta: float | NumberList, pow: float, inplace:bool):
    """debias 2nd momentum, optionally in-place"""
    if inplace: return tensors_.div_(bias_correction2_value(step, beta=beta, pow=pow))
    return tensors_ / bias_correction2_value(step, beta=beta, pow=pow)


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
    else:
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

    if debiased: sqrt_exp_avg_sq = debias2(sqrt_exp_avg_sq, step=step, beta=beta, pow=pow, inplace=True)
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


