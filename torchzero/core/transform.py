from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Literal

import torch

from .module import Module, Vars

Target = Literal['grad', 'update', 'closure', 'params_direct', 'params_difference', 'update_difference']

class Transform(Module, ABC):
    def __init__(self, defaults: dict[str,Any] | None = None, target: Target = 'update'):
        super().__init__(defaults)
        self._target: Target = target

    @abstractmethod
    def transform(self, target: list[torch.Tensor], vars: Vars) -> Iterable[torch.Tensor]:
        """applies the update rule to `target`"""

    def step(self, vars: Vars) -> Vars:
        # ---------------------------------- update ---------------------------------- #
        if self._target == 'update':
            vars.update = list(self.transform(vars.get_update(), vars))
            return vars

        # ----------------------------------- grad ----------------------------------- #
        if self._target == 'grad':
            vars.grad = list(self.transform(vars.get_grad(), vars))
            return vars

        # ------------------------------- params_direct ------------------------------ #
        if self._target == 'params_direct':
            new_params = self.transform(vars.params, vars)
            for p, new_p in zip(vars.params, new_params): p.set_(new_p) # pyright: ignore[reportArgumentType]
            return vars

        # ----------------------------- params_differnce ----------------------------- #
        if self._target == 'params_difference':
            new_params = tuple(self.transform([p.clone() for p in vars.params], vars))
            vars.update = list(torch._foreach_sub(vars.params, new_params))
            return vars

        # ----------------------------- update_difference ---------------------------- #
        if self._target == 'update_difference':
            update = vars.get_update()
            new_update = tuple(self.transform([u.clone() for u in update], vars))
            vars.update = list(torch._foreach_sub(update, new_update))
            return vars

        # ---------------------------------- closure --------------------------------- #
        if self._target == 'closure':
            original_closure = vars.closure
            if original_closure is None: raise ValueError('Target = "closure", but closure is None')

            params = vars.params
            def transformed_closure(backward=True):
                if backward:
                    loss = original_closure()
                    grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                    transformed_grad = list(self.transform(grad, vars))
                    for p, g in zip(params, transformed_grad):
                        p.grad = g

                else:
                    loss = original_closure(False)

                return loss

            vars.closure = transformed_closure
            return vars

        # ---------------------------------- invalid --------------------------------- #
        raise ValueError(f'Invalid target: {self._target}')


class ParameterwiseTransform(Module, ABC):
    def __init__(self, requires_grad: bool, defaults: dict[str,Any] | None = None, target: Target = 'update'):
        super().__init__(defaults)
        self._target: Target = target
        self._requires_grad: bool = requires_grad

    @abstractmethod
    def transform(
        self,
        target: torch.Tensor,
        param: torch.Tensor,
        grad: torch.Tensor | None,
        vars: Vars,
    ) -> torch.Tensor:
        """applies the update rule to `target`"""

    def step(self, vars: Vars) -> Vars:
        params = vars.params
        if self._requires_grad and vars.grad is None: vars.get_grad()

        # ---------------------------------- update ---------------------------------- #
        if self._target == 'update':
            update = vars.get_update()
            grad = vars.grad if vars.grad is not None else [None] * len(params)
            transformed_update = []

            for i, (p, g, u) in enumerate(zip(params, grad, update)):
                transformed_update.append(self.transform(target=u, param=p, grad=g, vars=vars))

            vars.update = transformed_update
            return vars

        # ----------------------------------- grad ----------------------------------- #
        if self._target == 'grad':
            grad = vars.get_grad()
            transformed_grad = []

            for i, (p, g) in enumerate(zip(params, grad)):
                transformed_grad.append(self.transform(target=g, param=p, grad=g, vars=vars))

            vars.grad = transformed_grad
            return vars

        # ------------------------------- params_direct ------------------------------ #
        if self._target == 'params_direct':
            grad = vars.grad if vars.grad is not None else [None] * len(params)

            for i, (p, g) in enumerate(zip(params, grad)):
                p.set_(self.transform(target=p, param=p, grad=g, vars=vars)) # type:ignore

            return vars

        # ----------------------------- params_difference ---------------------------- #
        if self._target == 'params_difference':
            grad = vars.grad if vars.grad is not None else [None] * len(params)
            transformed_params = []

            for i, (p, g) in enumerate(zip(params, grad)):
                transformed_params.append(self.transform(target=p.clone(), param=p, grad=g, vars=vars))

            vars.update = list(torch._foreach_sub(params, transformed_params))
            return vars

        # ----------------------------- update_difference ---------------------------- #
        if self._target == 'update_difference':
            update = vars.get_update()
            grad = vars.grad if vars.grad is not None else [None] * len(params)
            transformed_update = []

            for i, (p, g, u) in enumerate(zip(params, grad, update)):
                transformed_update.append(self.transform(target=u.clone(), param=p, grad=g, vars=vars))

            vars.update = list(torch._foreach_sub(update, transformed_update))
            return vars

        # ---------------------------------- closure --------------------------------- #
        if self._target == 'closure':
            original_closure = vars.closure
            if original_closure is None: raise ValueError('Target = "closure", but closure is None')

            params = vars.params
            def transformed_closure(backward=True):
                if backward:
                    loss = original_closure()
                    grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                    transformed_grad = []

                    for i, (p, g) in enumerate(zip(params, grad)):
                        transformed_grad.append(self.transform(target=g, param=p, grad=g, vars=vars))

                    for p, g in zip(params, transformed_grad):
                        p.grad = g

                else:
                    loss = original_closure(False)

                return loss

            vars.closure = transformed_closure
            return vars

        # ---------------------------------- invalid --------------------------------- #
        raise ValueError(f'Invalid target: {self._target}')