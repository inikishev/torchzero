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
from ..functional import add_power_, lerp_power_, root


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

    # inner args
    inner: Module | None = None,
    params: list[torch.Tensor] | None = None,
    grads: list[torch.Tensor] | None = None,
):
    """returns `tensors_`"""
    clr = alpha / (1 + step * lr_decay)

    sq_sum_ = add_power_(tensors_, sum_=sq_sum_, pow=pow)

    if inner is not None:
        assert params is not None
        tensors_ = TensorList(apply_transform(inner, tensors_, params=params, grads=grads))

    if divide: sq_sum_ = sq_sum_ / max(step, 1)

    if use_sqrt: tensors_.div_(root(sq_sum_, p=pow, inplace=False).add_(eps)).mul_(clr)
    else: tensors_.div_(sq_sum_.add(eps)).mul_(clr)

    return tensors_



class Adagrad(Transform):
    """Adagrad, divides by sum of past squares of gradients.

    This implementation is identical to :code:`torch.optim.Adagrad`.

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
        inner: Chainable | None = None,
    ):
        defaults = dict(alpha = alpha, lr_decay = lr_decay, initial_accumulator_value=initial_accumulator_value,
                        eps = eps, pow=pow, use_sqrt = use_sqrt, divide=divide)
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
            step=self.global_state["step"],
            pow=pow,
            use_sqrt=use_sqrt,
            divide=divide,

            # inner args
            inner=self.children.get("inner", None),
            params=params,
            grads=grads,
        )



class FullMatrixAdagrad(TensorwiseTransform):
    def __init__(self, beta: float | None = None, decay: float | None = None, sqrt:bool=True, concat_params=True, update_freq=1, init: Literal['identity', 'zeros', 'ones', 'GGT'] = 'identity', divide: bool=False, inner: Chainable | None = None):
        defaults = dict(beta=beta, decay=decay, sqrt=sqrt, init=init, divide=divide)
        super().__init__(defaults, uses_grad=False, concat_params=concat_params, update_freq=update_freq, inner=inner,)

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
        GG = state['GG']
        sqrt = setting['sqrt']
        divide = setting['divide']
        if divide: GG = GG/state.get('i', 1)

        if tensor.numel() == 1:
            GG = GG.squeeze()
            if sqrt: return tensor / GG.sqrt()
            return tensor / GG

        try:
            if sqrt: B = matrix_power_eigh(GG, -1/2)
            else: return torch.linalg.solve(GG, tensor.ravel()).view_as(tensor) # pylint:disable = not-callable

        except torch.linalg.LinAlgError:
            scale = 1 / tensor.abs().max()
            return tensor.mul_(scale.clip(min=torch.finfo(tensor.dtype).eps, max=1)) # conservative scaling

        return (B @ tensor.ravel()).view_as(tensor)

