from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence, Mapping
from typing import Any, Literal, final

import torch

from ..utils import set_storage_, TensorList, vec_to_tensors
from .module import Module, Var, Chain, Chainable

Target = Literal['grad', 'update', 'closure', 'params_direct', 'params_difference', 'update_difference']

class Transform(Module, ABC):
    """Base class for a transform. This is an abstract class, to use it, subclass it and override `update` and `apply` methods.

    A transform is a module that can also be applied manually to an arbitrary sequence of tensors.

    Args:
        defaults (dict[str,Any] | None): dict with default values.
        uses_grad (bool):
            Set this to True if `transform` method uses the `grad` argument. This will ensure
            `grad` is always computed and can't be None. Otherwise set to False.
        target (Target, optional):
            what to set on var. Defaults to 'update'.
    """
    def __init__(
        self,
        defaults: dict[str,Any] | None,
        uses_grad: bool,
        concat_params: bool = False,
        update_freq: int = 1,
        scale_first: bool = False,
        inner: Chainable | None = None,
        target: Target = 'update',
    ):
        super().__init__(defaults)
        self._target: Target = target
        self._uses_grad = uses_grad
        self._concat_params = concat_params
        self._update_freq = update_freq
        self._scale_first = scale_first
        self._inner = inner

    def update(
        self,
        tensors: list[torch.Tensor],
        params: list[torch.Tensor],
        grads: list[torch.Tensor] | None,
        loss: torch.Tensor | None,
        states: list[dict[str, Any]],
        settings: Sequence[Mapping[str, Any]],
    ) -> None:
        """Updates this transform. By default does nothing - if logic is in `apply` method."""

    @abstractmethod
    def apply(
        self,
        tensors: list[torch.Tensor],
        params: list[torch.Tensor],
        grads: list[torch.Tensor] | None,
        loss: torch.Tensor | None,
        states: list[dict[str, Any]],
        settings: Sequence[Mapping[str, Any]],
    ) -> Sequence[torch.Tensor]:
        """Applies the update rule to `tensors`."""

    @final
    @torch.no_grad
    def transform(
        self,
        tensors: list[torch.Tensor],
        params: list[torch.Tensor],
        grads: list[torch.Tensor] | None,
        loss: torch.Tensor | None,
        states: list[dict[str, Any]],
        settings: Sequence[Mapping[str, Any]] | None,
    ) -> list[torch.Tensor]:
        """Applies this transform to an arbitrary sequence of tensors."""
        un_tensors = tensors
        un_params = params
        un_grads = grads
        if self._concat_params:
            tensors = [torch.cat([t.ravel() for t in tensors])]
            params = [torch.cat([p.ravel() for p in params])]
            grads = [torch.cat([g.ravel() for g in grads])] if grads is not None else None

        if settings is None:
            settings = [self.defaults for _ in tensors]

        step = self.global_state.get('__step', 0)
        num = len(tensors)
        states = states[:num]
        settings = settings[:num]

        update_freq = self._update_freq
        scale_first = self._scale_first
        scale_factor = 1

        # scaling factor for 1st step
        if scale_first and step == 0:
            # initial step size guess from pytorch LBFGS
            scale_factor = 1 / TensorList(tensors).abs().global_sum().clip(min=1)
            scale_factor = scale_factor.clip(min=torch.finfo(tensors[0].dtype).eps)

        # update transform
        if step % update_freq == 0:
            self.update(tensors=tensors, params=params, grads=grads, loss=loss, states=states, settings=settings)

        # step with inner
        if self._inner is not None:
            tensors = apply_transform(self._inner, tensors=un_tensors, params=un_params, grads=un_grads)
            if self._concat_params:
                tensors = [torch.cat([t.ravel() for t in tensors])]

        # apply transform
        tensors = list(self.apply(tensors=tensors, params=params, grads=grads, loss=loss, states=states, settings=settings))

        # scale initial step, when preconditioner might not have been applied
        if scale_first and step == 0:
            torch._foreach_mul_(tensors, scale_factor)

        self.global_state['__step'] = step + 1
        if self._concat_params:
            tensors = vec_to_tensors(vec=tensors[0], reference=un_tensors)
        return tensors


    @torch.no_grad
    def keyed_transform(
        self,
        tensors: list[torch.Tensor],
        params: list[torch.Tensor],
        grads: list[torch.Tensor] | None,
        loss: torch.Tensor | None,
    ):
        """Applies this transform to `tensors`, `params` will be used as keys and need to always point to same tensor objects."""
        if self._concat_params:
            p = params[0]
            states = [self.state[p]]
            settings = [self.settings[p]]

        else:
            states = []
            settings = []
            for p in params:
                states.append(self.state[p])
                settings.append(self.settings[p])

        return self.transform(tensors=tensors, params=params, grads=grads, loss=loss, states=states, settings=settings)

    def step(self, var: Var) -> Var:
        # var may change, therefore current params and grads have to be extracted and passed explicitly
        if self._uses_grad: var.get_grad()
        params=var.params

        # ---------------------------------- update ---------------------------------- #
        if self._target == 'update':
            update = var.get_update()
            var.update = list(self.keyed_transform(update, params, var.grad, var.loss))
            return var

        # ----------------------------------- grad ----------------------------------- #
        if self._target == 'grad':
            grad = var.get_grad()
            var.grad = list(self.keyed_transform(grad, params, grad, var.loss))
            return var

        # ------------------------------- params_direct ------------------------------ #
        if self._target == 'params_direct':
            new_params = self.keyed_transform(var.params, params, var.grad, var.loss)
            for p, new_p in zip(var.params, new_params): set_storage_(p, new_p)
            return var

        # ----------------------------- params_differnce ----------------------------- #
        if self._target == 'params_difference':
            new_params = tuple(self.keyed_transform([p.clone() for p in var.params], params, var.grad, var.loss))
            var.update = list(torch._foreach_sub(var.params, new_params))
            return var

        # ----------------------------- update_difference ---------------------------- #
        if self._target == 'update_difference':
            update = var.get_update()
            new_update = tuple(self.keyed_transform([u.clone() for u in update], params, var.grad, var.loss))
            var.update = list(torch._foreach_sub(update, new_update))
            return var

        # ---------------------------------- closure --------------------------------- #
        if self._target == 'closure':
            original_closure = var.closure
            if original_closure is None: raise ValueError('Target = "closure", but closure is None')

            params = var.params
            def transformed_closure(backward=True):
                if backward:
                    loss = original_closure()
                    current_grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                    transformed_grad = list(self.keyed_transform(current_grad, params, var.grad, var.loss))
                    for p, g in zip(params, transformed_grad):
                        p.grad = g

                else:
                    loss = original_closure(False)

                return loss

            var.closure = transformed_closure
            return var

        # ---------------------------------- invalid --------------------------------- #
        raise ValueError(f'Invalid target: {self._target}')


class TensorwiseTransform(Transform, ABC):
    """Base class for a parameter-wise transform.

    This is an abstract class, to use it, subclass it and override `transform`.

    Args:
        defaults (dict[str,Any] | None): dict with default values.
        uses_grad (bool):
            Set this to True if `transform` method uses the `grad` argument. This will ensure
            `grad` is always computed and can't be None. Otherwise set to False.
        target (Target, optional):
            what to set on var. Defaults to 'update'.
    """
    def __init__(
        self,
        defaults: dict[str,Any] | None,
        uses_grad: bool,
        concat_params: bool = False,
        update_freq: int = 1,
        scale_first: bool = False,
        inner: Chainable | None = None,
        target: Target = 'update',
    ):
        super().__init__(
            defaults=defaults,
            uses_grad=uses_grad,
            concat_params=concat_params,
            update_freq=update_freq,
            scale_first=scale_first,
            inner=inner,
            target=target,
        )

    def update_tensor(
        self,
        tensor: torch.Tensor,
        param: torch.Tensor,
        grad: torch.Tensor | None,
        loss: torch.Tensor | None,
        state: dict[str, Any],
        settings: Mapping[str, Any],
    ) -> None:
        """Updates this transform. By default does nothing - if logic is in `apply` method."""

    @abstractmethod
    def apply_tensor(
        self,
        tensor: torch.Tensor,
        param: torch.Tensor,
        grad: torch.Tensor | None,
        loss: torch.Tensor | None,
        state: dict[str, Any],
        settings: Mapping[str, Any],
    ) -> torch.Tensor:
        """Applies the update rule to `tensor`."""

    @final
    def update(self, tensors, params, grads, loss, states, settings):
        if grads is None: grads = [None]*len(tensors)
        for t,p,g,state,setting in zip(tensors, params, grads, states, settings):
            self.update_tensor(t, p, g, loss, state, setting)

    @final
    def apply(self, tensors, params, grads, loss, states, settings):
        applied = []
        if grads is None: grads = [None]*len(tensors)
        for t,p,g,state,setting in zip(tensors, params, grads, states, settings):
            applied.append(self.apply_tensor(t, p, g, loss, state, setting))
        return applied

def apply_transform(
    tfm: Chainable,
    tensors: list[torch.Tensor],
    params: list[torch.Tensor],
    grads: list[torch.Tensor] | None,
    loss: torch.Tensor | None = None,
    var: Var | None = None,
    current_step: int = 0,
):
    if var is None:
        var = Var(params=params, closure=None, model=None, current_step=current_step)
        var.loss = loss

    if isinstance(tfm, Transform):
        if tfm._uses_grad and grads is None: grads = var.get_grad()
        return list(tfm.keyed_transform(tensors, params, grads, loss))

    if isinstance(tfm, Chain): tfm = tfm.get_children_sequence() # pyright: ignore[reportAssignmentType]
    if isinstance(tfm, Sequence):
        for module in tfm:
            tensors = apply_transform(module, tensors=tensors, params=params, grads=grads, var=var)
        return tensors

    if isinstance(tfm, Module):
        cvar = var.clone(clone_update=False)
        cvar.update = tensors
        cvar = tfm.step(cvar)
        var.update_attrs_from_clone_(cvar)
        assert cvar.update is not None
        return cvar.update

    raise TypeError(type(tfm))