import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

import torch

from ...core import Module, Vars
from ...utils import NumberList, TensorList
from ...utils.derivatives import jacobian_wrt
from ..grad_approximation import GradApproximator, GradTarget


class Reformulation(Module, ABC):
    def __init__(self, defaults):
        super().__init__(defaults)

    @abstractmethod
    def closure(self, backward: bool, closure: Callable, params:list[torch.Tensor], vars: Vars) -> tuple[float | torch.Tensor, Sequence[torch.Tensor] | None]:
        """returns loss and gradient, if backward is False then gradient can be None"""

    def pre_step(self, vars: Vars) -> Vars | None:
        """This runs once before each step, whereas `closure` may run multiple times per step if further modules
        evaluate gradients at multiple points. This is useful for example to pre-generate new random perturbations."""
        return vars

    def step(self, vars):
        ret = self.pre_step(vars)
        if isinstance(ret, Vars): vars = ret

        if vars.closure is None: raise RuntimeError("Reformulation requires closure")
        params, closure = vars.params, vars.closure


        def modified_closure(backward=True):
            loss, grad = self.closure(backward, closure, params, vars)

            if grad is not None:
                for p,g in zip(params, grad):
                    p.grad = g

            return loss

        vars.closure = modified_closure
        return vars


class GaussianHomotopy(Reformulation):
    def __init__(self, init_sigma: float, tol=1e-4, decay=0.5, max_steps:int| None=None, beta: float | None = None):
        defaults = dict(init_sigma=init_sigma, tol=tol, decay=decay, max_steps=max_steps, beta=beta)
        super().__init__(defaults)

    def pre_step(self, vars):
        beta = self.settings[vars.params[0]]['beta']
        if beta is not None:
            if 'perturbations' not in self.global_state: 1

    def closure(self, backward, closure, params, vars):

        1