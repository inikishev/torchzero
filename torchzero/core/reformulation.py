from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch

from .module import Chainable, Modular, Module, Var


class Reformulation(Module, ABC):
    def __init__(self, defaults: dict | None, modules: Chainable | None):
        super().__init__(defaults)

        if modules is not None:
            self.set_child("modules", modules)

    @abstractmethod
    def closure(self, backward: bool, closure: Callable, params:list[torch.Tensor], var: Var) -> tuple[float | torch.Tensor, Sequence[torch.Tensor] | None]:
        """
        returns (loss, gradient), if backward is False then gradient can be None.

        If evaluating original loss/gradient at x_0, set them to ``var``.
        """

    def pre_step(self, var: Var) -> Var | None:
        """This runs once before each step, whereas `closure` may run multiple times per step if further modules
        evaluate gradients at multiple points. This is useful for example to pre-generate new random perturbations."""

    def step(self, var):
        ret = self.pre_step(var) # pylint:disable = assignment-from-no-return
        if isinstance(ret, Var): var = ret

        if var.closure is None: raise RuntimeError("Reformulation requires closure")
        params, closure = var.params, var.closure

        # step with children
        if 'modules' in self.children:

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

            # step with child
            modules = self.children['modules']
            modified_var = modules.step(modified_var)

            # modified_var.loss and grad refers to loss and grad of a modified objective
            # so we only take the update
            var.update = modified_var.update

        # or just evaluate new closure and set to update
        else:
            loss, grad = self.closure(backward=True, closure=closure, params=params, var=var)
            if grad is not None: var.update = list(grad)

        return var
