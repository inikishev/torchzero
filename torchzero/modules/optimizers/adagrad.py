from operator import itemgetter

import torch

from ...core import (
    Chainable,
    Module,
    Preconditioner,
    Target,
    TensorwisePreconditioner,
    Transform,
    Vars,
    apply,
)
from ...utils import NumberList, TensorList
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

    # inner args
    inner: Module | None = None,
    params: list[torch.Tensor] | None = None,
    grads: list[torch.Tensor] | None = None,
    vars: Vars | None = None,
):
    """returns `tensors_`"""
    clr = alpha / (1 + step * lr_decay)

    sq_sum_ = add_power_(tensors_, sum_=sq_sum_, pow=pow)

    if inner is not None:
        assert params is not None
        tensors_ = TensorList(apply(inner, tensors_, params=params, grads=grads, vars=vars))

    if use_sqrt: tensors_.div_(root(sq_sum_, p=pow, inplace=False).add_(eps)).mul_(clr)
    else: tensors_.div_(sq_sum_.add(eps)).mul_(clr)

    return tensors_



class Adagrad(Transform):
    """Adagrad, divides by sum of past squares of gradients, matches pytorch Adagrad.

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
        inner: Chainable | None = None,
    ):
        defaults = dict(alpha = alpha, lr_decay = lr_decay, initial_accumulator_value=initial_accumulator_value,
                        eps = eps, pow=pow, use_sqrt = use_sqrt)
        super().__init__(defaults=defaults, uses_grad=False)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        tensors = TensorList(tensors)
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        lr_decay,alpha,eps = self.get_settings('lr_decay', 'alpha', 'eps', params=params, cls=NumberList)

        pow, use_sqrt = itemgetter('pow', 'use_sqrt')(self.settings[params[0]])

        sq_sum = self.get_state('sq_sum', params=params, cls=TensorList)

        # initialize accumulator on 1st step
        if step == 1:
            sq_sum.set_(tensors.full_like(self.get_settings('initial_accumulator_value', params=params)))

        return adagrad_(
            tensors,
            sq_sum_=sq_sum,
            alpha=alpha,
            lr_decay=lr_decay,
            eps=eps,
            step=self.global_state["step"],
            pow=pow,
            use_sqrt=use_sqrt,

            # inner args
            inner=self.children.get("inner", None),
            params=params,
            grads=grads,
            vars=vars,
        )



class FullMatrixAdagrad(TensorwisePreconditioner):
    def __init__(self, eps: float = 1e-8, beta: float | None = None, decay: float | None = None, concat_params=False, update_freq=1, inner: Chainable | None = None):
        defaults = dict(eps=eps, beta=beta, decay=decay)
        super().__init__(defaults, uses_grad=False, concat_params=concat_params, update_freq=update_freq, inner=inner)

    @torch.no_grad
    def update_tensor(self, tensor, param, grad, state, settings):
        G = tensor.ravel()
        GG = torch.outer(G, G)
        decay = settings['decay']
        beta = settings['beta']

        if 'GG' not in state: state['GG'] = torch.eye(GG.size(0), device=GG.device, dtype=GG.dtype)
        if decay is not None: state['GG'].mul_(decay)

        if beta is not None: state['GG'].lerp_(GG, 1-beta)
        else: state['GG'].add_(GG)

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, state, settings):
        GG = state['GG']
        eps = settings['eps']

        if tensor.numel() == 1:
            return tensor / (GG**(1/2)).squeeze()

        try:
            B = matrix_power_eigh(GG, -1/2, eps=eps)
        except torch.linalg.LinAlgError:
            return tensor.div_(tensor.abs().max()) # conservative scaling

        return (B @ tensor.ravel()).view_as(tensor)

