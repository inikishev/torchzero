import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, Literal

import torch

from ...core import Module, Vars

GradTarget = Literal['update', 'grad', 'closure']
_Scalar = torch.Tensor | float

class GradApproximator(Module, ABC):
    """Base class for gradient approximations.
    This is an abstract class, to use it, subclass it and override `approximate`.

    Args:
        defaults (dict[str, Any] | None, optional): dict with defaults. Defaults to None.
        target (str, optional):
            whether to set `vars.grad`, `vars.update` or 'vars.closure`. Defaults to 'closure'.
    """
    def __init__(self, defaults: dict[str, Any] | None = None, target: GradTarget = 'closure'):
        super().__init__(defaults)
        self._target: GradTarget = target

    @abstractmethod
    def approximate(self, closure: Callable, params: list[torch.Tensor], loss: _Scalar | None, vars: Vars) -> tuple[Iterable[torch.Tensor], _Scalar | None, _Scalar | None]:
        """Returns a tuple: (grad, loss, loss_approx), make sure this resets parameters to their original values!"""

    def pre_step(self, vars: Vars) -> Vars | None:
        """This runs once before each step, whereas `approximate` may run multiple times per step if further modules
        evaluate gradients at multiple points. This is useful for example to pre-generate new random perturbations."""
        return vars

    @torch.no_grad
    def step(self, vars):
        ret = self.pre_step(vars)
        if isinstance(ret, Vars): vars = ret

        if vars.closure is None: raise RuntimeError("Gradient approximation requires closure")
        params, closure, loss = vars.params, vars.closure, vars.loss

        if self._target == 'closure':

            def approx_closure(backward=True):
                if backward:
                    grad, l, l_approx = self.approximate(closure=closure, params=params, loss=loss, vars=vars)
                    for p, g in zip(params, grad): p.grad = g
                    return l if l is not None else l_approx
                return closure(False)

            vars.closure = approx_closure
            return vars

        # if vars.grad is not None:
        #     warnings.warn('Using grad approximator when `vars.grad` is already set.')
        grad,loss,loss_approx = self.approximate(closure=closure, params=params, loss=loss, vars=vars)
        if loss_approx is not None: vars.loss_approx = loss_approx
        if loss is not None: vars.loss = vars.loss_approx = loss
        if self._target == 'grad': vars.grad = list(grad)
        elif self._target == 'update': vars.update = list(grad)
        else: raise ValueError(self._target)
        return vars

_FD_Formula = Literal['forward2', 'backward2', 'forward3', 'backward3', 'central2', 'central4']