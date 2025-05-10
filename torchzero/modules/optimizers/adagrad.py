from operator import itemgetter

import torch

from ...core import Chainable, Module, Target, Transform, Vars, apply
from ...utils import NumberList, TensorList
from ..functional import add_power_, lerp_power_, root


def adagrad_(
    tensors_: TensorList,
    sq_sum_: TensorList,
    alpha: float | NumberList,
    lr_decay: float | NumberList,
    eps: float | NumberList,
    beta: float | NumberList | None,
    decay: float | NumberList,
    step: int,
    pow: float = 2,
    use_sqrt: bool = True,

    # inner args
    inner: Module | None = None,
    params: list[torch.Tensor] | None = None,
    grads: list[torch.Tensor] | None = None,
    vars: Vars | None = None,
):
    """returns `tensors_`"""
    clr = alpha / (1 + step * lr_decay)

    if beta is None or (isinstance(beta, NumberList) and beta[0] is None):
        sq_sum_ = add_power_(tensors_, sum_=sq_sum_, pow=pow)
    else:
        sq_sum_ = lerp_power_(tensors_, exp_avg_pow_=sq_sum_, beta=beta, pow=pow)

    if inner is not None:
        assert params is not None
        tensors_ = TensorList(apply(inner, tensors_, params=params, grads=grads, vars=vars))

    if use_sqrt: tensors_.div_(root(sq_sum_, p=pow, inplace=False).add_(eps)).mul_(clr)
    else: tensors_.div_(sq_sum_.add(eps)).mul_(clr)

    sq_sum_.lazy_mul_(decay)
    return tensors_



class Adagrad(Transform):
    def __init__(
        self,
        lr_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        alpha: float = 1,
        beta: float | None = None,
        decay: float = 1,
        pow: float = 2,
        use_sqrt: bool = True,
        target: Target = 'update',
        inner: Chainable | None = None,
    ):
        defaults = dict(alpha = alpha, lr_decay = lr_decay, initial_accumulator_value=initial_accumulator_value,
                        eps = eps, beta=beta, decay = decay, pow=pow, use_sqrt = use_sqrt)
        super().__init__(defaults=defaults, uses_grad=False, target=target)

        if inner is not None:
            self.set_child('inner', inner)

    def transform(self, tensors, params, grads, vars):
        tensors = TensorList(tensors)
        self.counter.increment()

        lr_decay,alpha,eps,beta,decay  = self.get_settings('lr_decay', 'alpha', 'eps', 'beta', 'decay', params=params, cls=NumberList)

        pow, use_sqrt = itemgetter('pow', 'use_sqrt')(self.settings[params[0]])

        sq_sum = self.get_state('sq_sum', params=params, cls=TensorList)

        # initialize accumulator on 1st step
        if self.counter() == 1:
            sq_sum.set_(tensors.full_like(self.get_settings('initial_accumulator_value', params=params)))

        return adagrad_(
            tensors,
            sq_sum_=sq_sum,
            alpha=alpha,
            lr_decay=lr_decay,
            eps=eps,
            beta=beta,
            decay=decay,
            step=self.global_state["step"],
            pow=pow,
            use_sqrt=use_sqrt,

            # inner args
            inner=self.children.get("inner", None),
            params=params,
            grads=grads,
            vars=vars,
        )