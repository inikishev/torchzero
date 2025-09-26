from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch

from .module import Chainable, Module
from .objective import Objective


class Reformulation(Module, ABC):
    """Reformulation allows the definition of a new closure which returns custom loss and gradient.

    If ``modules`` are passed, steps with those modules using the reformulated closure. Only ``step`` method is supported.

    If ``modules`` is ``None``, sets new closure to the objective so that all further modules use it.
    In that case make sure this method is first.

    To use this, subclass and override ``closure`` and optionally ``pre_step``.
    """
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

    def pre_step(self, objective: Objective):
        """This runs once before each step, whereas ``closure`` may run multiple times per step if further modules
        evaluate gradients at multiple points. This is useful for example to pre-generate new random perturbations."""

    def update(self, objective):
        if "modules" in self.children:
            raise RuntimeError("Reformulation ({self.__class__.__name__} only supports `step` method if it has sub-modules.)")

        self.pre_step(objective) # pylint:disable = assignment-from-no-return

        if objective.closure is None: raise RuntimeError("Reformulation requires closure")
        params, closure = objective.params, objective.closure # make sure to decouple from `objective` object

        # define modified closure and set objective to use it
        def modified_closure(backward=True):
            loss, grad = self.closure(backward, closure, params, objective)

            if grad is not None:
                for p,g in zip(params, grad):
                    p.grad = g

            return loss

        objective.closure = modified_closure

    def apply(self, objective): return objective

    def step(self, objective):

        if 'modules' in self.children:

            self.pre_step(objective) # pylint:disable = assignment-from-no-return

            if objective.closure is None: raise RuntimeError("Reformulation requires closure")
            params, closure = objective.params, objective.closure # make sure to decouple from `objective` object

            # make a reformulated closure
            def modified_closure(backward=True):
                loss, grad = self.closure(backward, closure, params, objective)

                if grad is not None:
                    for p,g in zip(params, grad):
                        p.grad = g

                return loss

            # set it to a new Objective object
            modified_objective = objective.clone(clone_updates=False)
            modified_objective.closure = modified_closure

            # update the child
            modules = self.children['modules']
            modified_objective = modules.step(modified_objective)

            # modified_var.loss and grad refers to loss and grad of a modified objective
            # so we only take the update
            objective.updates = modified_objective.updates

        # or just set closure to a modified one
        # update already calls self.pre_step
        else:
            self.update(objective)
            self.apply(objective) # does nothing unless overridden

        return objective
