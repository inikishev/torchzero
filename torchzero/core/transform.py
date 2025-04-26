from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import torch

from ..utils import set_storage_
from .module import Module, Vars, Chain, Chainable

Target = Literal['grad', 'update', 'closure', 'params_direct', 'params_difference', 'update_difference']

class Transform(Module, ABC):
    """Base class for a transform.

    This is an abstract class, to use it, subclass it and override `transform`.

    Args:
        defaults (dict[str,Any] | None): dict with default values.
        uses_grad (bool):
            Set this to True if `transform` method uses the `grad` argument. This will ensure
            `grad` is always computed and can't be None. Otherwise set to False.
        target (Target, optional):
            what to set on vars. Defaults to 'update'.
    """
    def __init__(self, defaults: dict[str,Any] | None, uses_grad: bool, target: Target = 'update'):
        super().__init__(defaults)
        self._target: Target = target
        self._uses_grad = uses_grad

    @abstractmethod
    def transform(self, target: list[torch.Tensor], params: list[torch.Tensor], grad: list[torch.Tensor] | None, vars: Vars) -> Iterable[torch.Tensor]:
        """applies the update rule to `target`."""

    def step(self, vars: Vars) -> Vars:
        # vars may change, therefore current params and grads have to be extracted and passed explicitly
        if self._uses_grad: vars.get_grad()
        params=vars.params; grad = vars.grad

        # ---------------------------------- update ---------------------------------- #
        if self._target == 'update':
            vars.update = list(self.transform(vars.get_update(), params, grad, vars))
            return vars

        # ----------------------------------- grad ----------------------------------- #
        if self._target == 'grad':
            vars.grad = list(self.transform(vars.get_grad(), params, grad, vars))
            return vars

        # ------------------------------- params_direct ------------------------------ #
        if self._target == 'params_direct':
            new_params = self.transform(vars.params, params, grad, vars)
            for p, new_p in zip(vars.params, new_params): set_storage_(p, new_p)
            return vars

        # ----------------------------- params_differnce ----------------------------- #
        if self._target == 'params_difference':
            new_params = tuple(self.transform([p.clone() for p in vars.params], params, grad, vars))
            vars.update = list(torch._foreach_sub(vars.params, new_params))
            return vars

        # ----------------------------- update_difference ---------------------------- #
        if self._target == 'update_difference':
            update = vars.get_update()
            new_update = tuple(self.transform([u.clone() for u in update], params, grad, vars))
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
                    current_grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                    transformed_grad = list(self.transform(current_grad, params, grad, vars))
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
    """Base class for a parameter-wise transform.

    This is an abstract class, to use it, subclass it and override `transform`.

    Args:
        defaults (dict[str,Any] | None): dict with default values.
        uses_grad (bool):
            Set this to True if `transform` method uses the `grad` argument. This will ensure
            `grad` is always computed and can't be None. Otherwise set to False.
        target (Target, optional):
            what to set on vars. Defaults to 'update'.
    """
    def __init__(self, defaults: dict[str,Any] | None, uses_grad: bool, target: Target = 'update'):
        super().__init__(defaults)
        self._target: Target = target
        self._uses_grad: bool = uses_grad

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
        if self._uses_grad and vars.grad is None: vars.get_grad()

        # ---------------------------------- update ---------------------------------- #
        if self._target == 'update':
            update = vars.get_update()
            grad = vars.grad if vars.grad is not None else [None] * len(params)
            transformed_update = []

            for p, g, u in zip(params, grad, update):
                # settings = self.settings[p] # couldn't make typing work with this
                #, self.transform(target=u, param=p, grad=g, vars=vars, **{k:settings[k] for k in self.defaults})
                transformed_update.append(self.transform(target=u, param=p, grad=g, vars=vars))

            vars.update = transformed_update
            return vars

        # ----------------------------------- grad ----------------------------------- #
        if self._target == 'grad':
            grad = vars.get_grad()
            transformed_grad = []

            for p, g in zip(params, grad):
                transformed_grad.append(self.transform(target=g, param=p, grad=g, vars=vars))

            vars.grad = transformed_grad
            return vars

        # ------------------------------- params_direct ------------------------------ #
        if self._target == 'params_direct':
            grad = vars.grad if vars.grad is not None else [None] * len(params)

            for p, g in zip(params, grad):
                set_storage_(p, self.transform(target=p, param=p, grad=g, vars=vars))

            return vars

        # ----------------------------- params_difference ---------------------------- #
        if self._target == 'params_difference':
            grad = vars.grad if vars.grad is not None else [None] * len(params)
            transformed_params = []

            for p, g in zip(params, grad):
                transformed_params.append(
                    self.transform(target=p.clone(), param=p, grad=g, vars=vars)
                )

            vars.update = list(torch._foreach_sub(params, transformed_params))
            return vars

        # ----------------------------- update_difference ---------------------------- #
        if self._target == 'update_difference':
            update = vars.get_update()
            grad = vars.grad if vars.grad is not None else [None] * len(params)
            transformed_update = []

            for p, g, u in zip(params, grad, update):
                transformed_update.append(
                    self.transform(target=u.clone(), param=p, grad=g, vars=vars)
                )

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

                    for p, g in zip(params, grad):
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



def apply(
    tfm: Chainable,
    target: list[torch.Tensor],
    params: list[torch.Tensor],
    grad: list[torch.Tensor] | None,
    vars: Vars | None = None,
    current_step: int = 0,
):
    if vars is None: vars = Vars(params=params, closure=None, model=None, current_step=current_step)
    if isinstance(tfm, Transform):
        if tfm._uses_grad and grad is None: grad = vars.get_grad()
        return list(tfm.transform(target, params, grad, vars))

    if isinstance(tfm, ParameterwiseTransform):
        grads_list = grad
        if grads_list is None:
            if tfm._uses_grad: grads_list = vars.get_grad()
            else: grads_list = [None] * len(target)
        return [tfm.transform(t, p, g, vars) for t,p,g in zip(target,params,grads_list)]

    if isinstance(tfm, Chain): tfm = tfm.get_children_sequence() # pyright: ignore[reportAssignmentType]
    if isinstance(tfm, Sequence):
        for module in tfm:
            target = apply(module, target=target, params=params, grad=grad, vars=vars)
        return target

    if isinstance(tfm, Module):
        cvars = vars.clone(clone_update=False)
        cvars.update = target
        cvars = tfm.step(cvars)
        vars.update_attrs_from_clone_(cvars)
        assert cvars.update is not None
        return cvars.update

    raise TypeError(type(tfm))