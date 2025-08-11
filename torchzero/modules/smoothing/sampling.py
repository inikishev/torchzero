import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal
import torch

from ...core import Modular, Module, Var, Chainable
from ...utils import NumberList, TensorList, Distributions


class Reformulation(Module, ABC):
    def __init__(self, defaults, modules: Chainable):
        super().__init__(defaults)

        self.set_child("modules", modules)

    @abstractmethod
    def closure(self, backward: bool, closure: Callable, params:list[torch.Tensor], var: Var) -> tuple[float | torch.Tensor, Sequence[torch.Tensor] | None]:
        """
        returns loss and gradient, if backward is False then gradient can be None.

        If evaluating original loss/gradient at x_0, set them to ``var``.
        """

    def pre_step(self, var: Var) -> Var | None:
        """This runs once before each step, whereas `closure` may run multiple times per step if further modules
        evaluate gradients at multiple points. This is useful for example to pre-generate new random perturbations."""

    def step(self, var):
        ret = self.pre_step(var)
        if isinstance(ret, Var): var = ret

        if var.closure is None: raise RuntimeError("Reformulation requires closure")
        params, closure = var.params, var.closure

        # make a reformulated closure
        def modified_closure(backward=True):
            loss, grad = self.closure(backward, closure, params, var)

            if grad is not None:
                for p,g in zip(params, grad):
                    p.grad = g

            return loss

        # set it to a new Var object
        modified_var = var.clone(clone_update=False)
        modified_var.closure = modified_closure

        # step with children
        modules = self.children['modules']
        modified_var = modules.step(modified_var)

        # modified_var.loss and grad refers to loss and grad of a modified objective
        # so we only take the update
        var.update = modified_var.update

        return var


class GradientSampling(Reformulation):
    def __init__(
        self,
        modules: Chainable,
        sigma: float = 1.,
        n:int = 100,
        aggregate = ...,
        distribution: Distributions = 'gaussian'
    ):
        defaults = dict(sigma=sigma, n=n, aggregate=aggregate, distribution=distribution)
        super().__init__(defaults, modules)

    def pre_step(self, var):
        # pre-generate perturnations
        params = TensorList(var.params)
        settings = [self.settings[p] for p in params]
        sigma_inits = [s['sigma'] for s in settings]

        states = [self.state[p] for p in params]
        sigmas = [s.setdefault('sigma', si) for s, si in zip(states, sigma_inits)]

        setting = self.settings[params[0]]
        n = setting['n']
        distribution = setting['distribution']

        perts = [params.sample_like(distribution)]
        