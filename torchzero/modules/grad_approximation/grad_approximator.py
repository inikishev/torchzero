import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, Literal

import torch

from ...core import Module, Var

GradTarget = Literal['update', 'grad', 'closure']
_Scalar = torch.Tensor | float

class GradApproximator(Module, ABC):
    """Base class for gradient approximations.
    This is an abstract class, to use it, subclass it and override `approximate`.

    Args:
        defaults (dict[str, Any] | None, optional): dict with defaults. Defaults to None.
        target (str, optional):
            whether to set `var.grad`, `var.update` or 'var.closure`. Defaults to 'closure'.
    """
    def __init__(self, defaults: dict[str, Any] | None = None, target: GradTarget = 'closure'):
        super().__init__(defaults)
        self._target: GradTarget = target

    @abstractmethod
    def approximate(self, closure: Callable, params: list[torch.Tensor], loss: _Scalar | None, var: Var) -> tuple[Iterable[torch.Tensor], _Scalar | None, _Scalar | None]:
        """Returns a tuple: (grad, loss, loss_approx), make sure this resets parameters to their original values!"""

    def pre_step(self, var: Var) -> Var | None:
        """This runs once before each step, whereas `approximate` may run multiple times per step if further modules
        evaluate gradients at multiple points. This is useful for example to pre-generate new random perturbations."""
        return var

    @torch.no_grad
    def step(self, var):
        ret = self.pre_step(var)
        if isinstance(ret, Var): var = ret

        if var.closure is None: raise RuntimeError("Gradient approximation requires closure")
        params, closure, loss = var.params, var.closure, var.loss

        if self._target == 'closure':

            def approx_closure(backward=True):
                if backward:
                    # set loss to None because closure might be evaluated at different points
                    grad, l, l_approx = self.approximate(closure=closure, params=params, loss=None, var=var)
                    for p, g in zip(params, grad): p.grad = g
                    return l if l is not None else l_approx
                return closure(False)

            var.closure = approx_closure
            return var

        # if var.grad is not None:
        #     warnings.warn('Using grad approximator when `var.grad` is already set.')
        grad,loss,loss_approx = self.approximate(closure=closure, params=params, loss=loss, var=var)
        if loss_approx is not None: var.loss_approx = loss_approx
        if loss is not None: var.loss = var.loss_approx = loss
        if self._target == 'grad': var.grad = list(grad)
        elif self._target == 'update': var.update = list(grad)
        else: raise ValueError(self._target)
        return var

_FD_Formula = Literal['forward2', 'backward2', 'forward3', 'backward3', 'central2', 'central4']