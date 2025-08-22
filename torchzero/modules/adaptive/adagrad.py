from operator import itemgetter
from typing import Literal

import torch
from ...core import (
    Chainable,
    Module,
    Target,
    TensorwiseTransform,
    Transform,
    Var,
    apply_transform,
)
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states
from ...utils.linalg import matrix_power_eigh
from ..functional import add_power_, lerp_power_, root, epsilon_step_size
from ...utils.linalg.linear_operator import Dense

def adagrad_(
    tensors_: TensorList,
    sq_sum_: TensorList,
    alpha: float | NumberList,
    lr_decay: float | NumberList,
    eps: float | NumberList,
    step: int,
    pow: float = 2,
    use_sqrt: bool = True,
    divide: bool = False,

    decay: float | None = None,
    beta: float | None = None,

    # inner args
    inner: Module | None = None,
    params: list[torch.Tensor] | None = None,
    grads: list[torch.Tensor] | None = None,
):
    """returns `tensors_`"""
    clr = alpha / (1 + step * lr_decay)

    if beta is None or step == 1: sq_sum_ = add_power_(tensors_, sum_=sq_sum_, pow=pow)
    else: sq_sum_ = lerp_power_(tensors_, exp_avg_pow_=sq_sum_, beta=beta, pow=pow)
    if decay is not None:
        sq_sum_.mul_(1-decay)

    if inner is not None:
        assert params is not None
        tensors_ = TensorList(apply_transform(inner, tensors_, params=params, grads=grads))

    if divide: sq_sum_ = sq_sum_ / max(step, 1)

    if use_sqrt: tensors_.div_(root(sq_sum_, p=pow, inplace=False).add_(eps)).mul_(clr)
    else: tensors_.div_(sq_sum_.add(eps)).mul_(clr)

    return tensors_



class Adagrad(Transform):
    """Adagrad, divides by sum of past squares of gradients.

    This implementation is identical to ``torch.optim.Adagrad``.

    Args:
        lr_decay (float, optional): learning rate decay. Defaults to 0.
        initial_accumulator_value (float, optional): initial value of the sum of squares of gradients. Defaults to 0.
        eps (float, optional): division epsilon. Defaults to 1e-10.
        alpha (float, optional): step size. Defaults to 1.
        pow (float, optional): power for gradients and accumulator root. Defaults to 2.
        use_sqrt (bool, optional): whether to take the root of the accumulator. Defaults to True.
        inner (Chainable | None, optional): Inner modules that are applied after updating accumulator and before preconditioning. Defaults to None.
    """
    def __init__(
        self,
        lr_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        alpha: float = 1,
        pow: float = 2,
        use_sqrt: bool = True,
        divide: bool=False,
        beta:float | None = None,
        decay: float | None = None,
        inner: Chainable | None = None,
    ):
        defaults = dict(alpha = alpha, lr_decay = lr_decay, initial_accumulator_value=initial_accumulator_value,
                        eps = eps, pow=pow, use_sqrt = use_sqrt, divide=divide, beta=beta, decay=decay)
        super().__init__(defaults=defaults, uses_grad=False)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        lr_decay,alpha,eps = unpack_dicts(settings, 'lr_decay', 'alpha', 'eps', cls=NumberList)

        pow, use_sqrt, divide = itemgetter('pow', 'use_sqrt', 'divide')(settings[0])

        sq_sum = unpack_states(states, tensors, 'sq_sum', cls=TensorList)

        # initialize accumulator on 1st step
        if step == 1:
            sq_sum.set_(tensors.full_like([s['initial_accumulator_value'] for s in settings]))

        return adagrad_(
            tensors,
            sq_sum_=sq_sum,
            alpha=alpha,
            lr_decay=lr_decay,
            eps=eps,
            step=step,
            pow=pow,
            use_sqrt=use_sqrt,
            divide=divide,

            beta = self.defaults["beta"],
            decay = self.defaults["decay"],
            # inner args
            inner=self.children.get("inner", None),
            params=params,
            grads=grads,
        )


def lerp(start, end, weight):
    return start + weight * (end - start)

def adagrad_norm_(
    tensors_: TensorList,
    accumulator: float | torch.Tensor,
    alpha: float | NumberList,
    lr_decay: float | NumberList,
    eps: float | NumberList,
    step: int,
    use_sqrt: bool = True,
    divide: bool = False,

    decay: float | None = None,
    beta: float | None = None,

    # inner args
    inner: Module | None = None,
    params: list[torch.Tensor] | None = None,
    grads: list[torch.Tensor] | None = None,
):
    """returns `tensors_`"""
    clr = alpha / (1 + step * lr_decay)

    gg = tensors_.dot(tensors_)

    if beta is None or step == 1: accumulator += gg
    else: accumulator = lerp(accumulator, gg, 1-beta)

    if decay is not None:
        accumulator *= 1-decay

    if inner is not None:
        assert params is not None
        tensors_ = TensorList(apply_transform(inner, tensors_, params=params, grads=grads))

    if divide: accumulator = accumulator / max(step, 1)

    if use_sqrt: tensors_.div_(eps + accumulator.sqrt()).mul_(clr)
    else: tensors_.div_(eps + accumulator).mul_(clr)

    return tensors_, accumulator

class AdagradNorm(Transform):
    """Adagrad-Norm, divides by sum of past means of squares of gradients.

    Args:
        lr_decay (float, optional): learning rate decay. Defaults to 0.
        initial_accumulator_value (float, optional): initial value of the sum of squares of gradients. Defaults to 0.
        eps (float, optional): division epsilon. Defaults to 1e-10.
        alpha (float, optional): step size. Defaults to 1.
        pow (float, optional): power for gradients and accumulator root. Defaults to 2.
        use_sqrt (bool, optional): whether to take the root of the accumulator. Defaults to True.
        inner (Chainable | None, optional): Inner modules that are applied after updating accumulator and before preconditioning. Defaults to None.
    """
    def __init__(
        self,
        lr_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        alpha: float = 1,
        pow: float = 2,
        use_sqrt: bool = True,
        divide: bool=False,
        beta:float | None = None,
        decay: float | None = None,
        inner: Chainable | None = None,
    ):
        defaults = dict(alpha = alpha, lr_decay = lr_decay, initial_accumulator_value=initial_accumulator_value,
                        eps = eps, pow=pow, use_sqrt = use_sqrt, divide=divide, beta=beta, decay=decay)
        super().__init__(defaults=defaults, uses_grad=False)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1
        lr_decay,alpha,eps = unpack_dicts(settings, 'lr_decay', 'alpha', 'eps', cls=NumberList)

        use_sqrt, divide, initial_accumulator_value = itemgetter('use_sqrt', 'divide', "initial_accumulator_value")(settings[0])

        accumulator = self.global_state.get("accumulator", initial_accumulator_value)

        d, self.global_state["accumulator"] = adagrad_norm_(
            tensors,
            accumulator=accumulator,
            alpha=alpha,
            lr_decay=lr_decay,
            eps=eps,
            step=step,
            use_sqrt=use_sqrt,
            divide=divide,

            beta = self.defaults["beta"],
            decay = self.defaults["decay"],
            # inner args
            inner=self.children.get("inner", None),
            params=params,
            grads=grads,
        )

        return d


class FullMatrixAdagrad(TensorwiseTransform):
    """Full-matrix version of Adagrad, can be customized to make RMSprop or Adam (see examples).

    Note:
        A more memory-efficient version equivalent to full matrix Adagrad on last n gradients is implemented in ``tz.m.LMAdagrad``.

    Args:
        beta (float | None, optional): momentum for gradient outer product accumulators. if None, uses sum. Defaults to None.
        decay (float | None, optional): decay for gradient outer product accumulators. Defaults to None.
        sqrt (bool, optional): whether to take the square root of the accumulator. Defaults to True.
        concat_params (bool, optional): if False, each parameter will have it's own accumulator. Defaults to True.
        precond_freq (int, optional): frequency of updating the inverse square root of the accumulator. Defaults to 1.
        init (Literal[str], optional):
            how to initialize the accumulator.
            - "identity" - with identity matrix (default).
            - "zeros" - with zero matrix.
            - "ones" - with matrix of ones.
             -"GGT" - with the first outer product
        divide (bool, optional): whether to divide the accumulator by number of gradients in it. Defaults to False.
        inner (Chainable | None, optional): inner modules to apply preconditioning to. Defaults to None.

    ## Examples:

    Plain full-matrix adagrad
    ```python
    opt = tz.Modular(
        model.parameters(),
        tz.m.FullMatrixAdagrd(),
        tz.m.LR(1e-2),
    )
    ```

    Full-matrix RMSprop
    ```python
    opt = tz.Modular(
        model.parameters(),
        tz.m.FullMatrixAdagrad(beta=0.99),
        tz.m.LR(1e-2),
    )
    ```

    Full-matrix Adam
    ```python
    opt = tz.Modular(
        model.parameters(),
        tz.m.FullMatrixAdagrad(beta=0.999, inner=tz.m.EMA(0.9)),
        tz.m.Debias(0.9, 0.999),
        tz.m.LR(1e-2),
    )
    ```
    """
    def __init__(
        self,
        beta: float | None = None,
        decay: float | None = None,
        sqrt: bool = True,
        concat_params=True,
        precond_freq: int = 1,
        init: Literal["identity", "zeros", "ones", "GGT"] = "identity",
        reg: float = 1e-12,
        divide: bool = False,
        inner: Chainable | None = None,
    ):
        defaults = dict(beta=beta, decay=decay, sqrt=sqrt, precond_freq=precond_freq, init=init, divide=divide, reg=reg)
        super().__init__(defaults, uses_grad=False, concat_params=concat_params, inner=inner,)

    @torch.no_grad
    def update_tensor(self, tensor, param, grad, loss, state, setting):
        G = tensor.ravel()
        GG = torch.outer(G, G)
        decay = setting['decay']
        beta = setting['beta']
        init = setting['init']

        if 'GG' not in state:
            if init == 'identity': state['GG'] = torch.eye(GG.size(0), device=GG.device, dtype=GG.dtype)
            elif init == 'zeros': state['GG'] =  torch.zeros_like(GG)
            elif init == 'ones': state['GG'] = torch.ones_like(GG)
            elif init == 'GGT': state['GG'] = GG.clone()
            else: raise ValueError(init)
        if decay is not None: state['GG'].mul_(decay)

        if beta is not None: state['GG'].lerp_(GG, 1-beta)
        else: state['GG'].add_(GG)
        state['i'] = state.get('i', 0) + 1 # number of GGTs in sum

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, loss, state, setting):
        step = state.get('step', 0)
        state['step'] = step + 1

        GG: torch.Tensor = state['GG']
        sqrt = setting['sqrt']
        divide = setting['divide']
        precond_freq = setting['precond_freq']
        reg = setting['reg']

        if divide: GG = GG/state.get('i', 1)

        if reg != 0:
            GG = GG + torch.eye(GG.size(0), device=GG.device, dtype=GG.dtype).mul_(reg)

        if tensor.numel() == 1:
            GG = GG.squeeze()
            if sqrt: return tensor / GG.sqrt()
            return tensor / GG

        try:
            if sqrt:
                if "B" not in state or step % precond_freq == 0:
                    B = state["B"] = matrix_power_eigh(GG, -1/2)
                else:
                    B = state["B"]

            else: return torch.linalg.solve(GG, tensor.ravel()).view_as(tensor) # pylint:disable = not-callable

        except torch.linalg.LinAlgError:
            # fallback to diagonal AdaGrad
            denom = GG.diagonal()
            if sqrt: denom = denom.sqrt()
            return tensor.div_(denom + max(reg, 1e-12))

        return (B @ tensor.ravel()).view_as(tensor)
