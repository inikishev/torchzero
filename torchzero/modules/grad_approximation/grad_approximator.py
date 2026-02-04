import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, Literal

import torch

from ...core import Module, Objective

GradTarget = Literal['update', 'grad', 'closure']
_Scalar = torch.Tensor | float

class GradApproximator(Module, ABC):
    """Base class for gradient approximations.
    This is an abstract class, to use it, subclass it and override `approximate`.

    GradientApproximator modifies the closure to evaluate the estimated gradients,
    and further closure-based modules will use the modified closure.

    Args:
        defaults (dict[str, Any] | None, optional): dict with defaults. Defaults to None.
        target (str, optional):
            whether to set `var.grad`, `var.update` or 'var.closure`. Defaults to 'closure'.

    Example:

    Basic SPSA method implementation.
    ```python
    class SPSA(GradApproximator):
        def __init__(self, h=1e-3):
            defaults = dict(h=h)
            super().__init__(defaults)

        @torch.no_grad
        def approximate(self, closure, params, loss):
            perturbation = [rademacher_like(p) * self.settings[p]['h'] for p in params]

            # evaluate params + perturbation
            torch._foreach_add_(params, perturbation)
            loss_plus = closure(False)

            # evaluate params - perturbation
            torch._foreach_sub_(params, perturbation)
            torch._foreach_sub_(params, perturbation)
            loss_minus = closure(False)

            # restore original params
            torch._foreach_add_(params, perturbation)

            # calculate SPSA gradients
            spsa_grads = []
            for p, pert in zip(params, perturbation):
                settings = self.settings[p]
                h = settings['h']
                d = (loss_plus - loss_minus) / (2*(h**2))
                spsa_grads.append(pert * d)

            # returns tuple: (grads, loss, loss_approx)
            # loss must be with initial parameters
            # since we only evaluated loss with perturbed parameters
            # we only have loss_approx
            return spsa_grads, None, loss_plus
    ```
    """
    def __init__(self, defaults: dict[str, Any] | None = None, return_approx_loss:bool=False, target: GradTarget = 'closure'):
        super().__init__(defaults)
        self._target: GradTarget = target
        self._return_approx_loss = return_approx_loss

    @abstractmethod
    def approximate(self, closure: Callable, params: list[torch.Tensor], loss: torch.Tensor | None) -> tuple[Iterable[torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
        """Returns a tuple: ``(grad, loss, loss_approx)``, make sure this resets parameters to their original values!"""

    def pre_step(self, objective: Objective) -> None:
        """This runs once before each step, whereas `approximate` may run multiple times per step if further modules
        evaluate gradients at multiple points. This is useful for example to pre-generate new random perturbations."""

    @torch.no_grad
    def update(self, objective):
        self.pre_step(objective)

        if objective.closure is None: raise RuntimeError("Gradient approximation requires closure")
        params, closure, loss = objective.params, objective.closure, objective.loss

        if self._target == 'closure':

            def approx_closure(backward=True):
                if backward:
                    # set loss to None because closure might be evaluated at different points
                    grad, l, l_approx = self.approximate(closure=closure, params=params, loss=None)
                    for p, g in zip(params, grad): p.grad = g
                    if l is not None: return l
                    if self._return_approx_loss and l_approx is not None: return l_approx
                    return closure(False)

                return closure(False)

            objective.closure = approx_closure
            return

        # if var.grad is not None:
        #     warnings.warn('Using grad approximator when `var.grad` is already set.')
        grad, loss, loss_approx = self.approximate(closure=closure, params=params, loss=loss)
        if loss_approx is not None: objective.loss_approx = loss_approx
        if loss is not None: objective.loss = objective.loss_approx = loss
        if self._target == 'grad': objective.grads = list(grad)
        elif self._target == 'update': objective.updates = list(grad)
        else: raise ValueError(self._target)
        return

    def apply(self, objective):
        return objective

_FD_Formula = Literal['forward', 'forward2', 'backward', 'backward2', 'central', 'central2', 'central3', 'forward3', 'backward3', 'central4', 'forward4', 'forward5', 'bspsa4']
