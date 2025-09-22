from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch

from .chain import Chain
from .module import Chainable, Module
from .objective import Objective


class Reformulation(Module, ABC):
    def __init__(self, defaults: dict | None, modules: Chainable | None):
        super().__init__(defaults)

        if modules is not None:
            self.set_child("modules", modules)

    @abstractmethod
    def closure(self, backward: bool, closure: Callable, params:list[torch.Tensor], objective: Objective) -> tuple[float | torch.Tensor, Sequence[torch.Tensor] | None]:
        """
        returns ``(loss, gradient)``, if backward is False then gradient can be None.

        If evaluating original loss/gradient at ``x0``, set them to ``objective``.
        """

    def pre_step(self, objective: Objective, states, settings):
        """This runs once before each step, whereas ``closure`` may run multiple times per step if further modules
        evaluate gradients at multiple points. This is useful for example to pre-generate new random perturbations."""

    def apply(self, objective, states, settings):
        self.pre_step(objective, states, settings) # pylint:disable = assignment-from-no-return

        if objective.closure is None: raise RuntimeError("Reformulation requires closure")
        params, closure = objective.params, objective.closure # make sure to decouple from `objective` object

        # step with children
        if 'modules' in self.children:

            # make a reformulated closure
            def modified_closure(backward=True):
                loss, grad = self.closure(backward, closure, params, objective)

                if grad is not None:
                    for p,g in zip(params, grad):
                        p.grad = g

                return loss

            # set it to a new Objective object
            modified_var = objective.clone(clone_update=False)
            modified_var.closure = modified_closure

            # step with child
            modules = self.children['modules']
            modified_var = modules.apply(modified_var)

            # modified_var.loss and grad refers to loss and grad of a modified objective
            # so we only take the update
            objective.update = modified_var.updates

        # or just evaluate new closure and set to update
        else:
            loss, grad = self.closure(backward=True, closure=closure, params=params, objective=objective)
            if grad is not None: objective.update = list(grad)

        return objective
