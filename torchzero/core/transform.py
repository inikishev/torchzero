from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from operator import itemgetter
from typing import Any, final, cast

import torch

from .chain import Chainable
from .module import Module
from ..utils import vec_to_tensors_, vec_to_tensors, safe_dict_update_
from .objective import Objective


class Transform(Module):
    """``Transform`` is a ``Module`` with no children except optional ``inner``.

    ``Transform`` if more flexible in that as long as ``inner=None``, it can use a custom list of states
    and settings instead of ``self.state`` and ``self.setting``.

    To use, subclass this and override ``update_states`` and ``apply_states``.
    """
    def __init__(self, defaults: dict[str, Any] | None = None, update_freq: int = 1, inner: Chainable | None = None):

        # store update_freq in defaults so that it is scheduleable
        if defaults is None: defaults = {}
        safe_dict_update_(defaults, {"update_freq": update_freq})

        super().__init__(defaults)

        if inner is not None:
            self.set_child("inner", inner)

    # settings shouldn't mutate, so they are typed as Sequence[Mapping]
    def update_states(self, objective: Objective, states: list[dict[str, Any]], settings: Sequence[Mapping[str, Any]]) -> None:
        """Updates ``states``. This should not modify ``objective.update``."""

    @abstractmethod
    def apply_states(self, objective: Objective, states: list[dict[str, Any]], settings: Sequence[Mapping[str, Any]]) -> Objective:
        """Updates ``objective`` using ``states``."""

    @final
    def update(self, objective:Objective):
        step = self.increment_counter("__step", 0)

        if step % self.defaults["update_freq"] == 0:
            states = itemgetter(objective.params)(self.state)
            settings = itemgetter(objective.params)(self.settings)
            self.update_states(objective=objective, states=states, settings=settings)

    @final
    def apply(self, objective: Objective):

        # inner step
        if "inner" in self.children:
            inner = self.children["inner"]

            inner.update(objective)
            objective = inner.apply(objective)

        # apply and return
        states = itemgetter(objective.params)(self.state)
        settings = itemgetter(objective.params)(self.settings)
        return self.apply_states(objective=objective, states=states, settings=settings)

class TensorTransform(Transform):
    """``TensorTransform`` is a ``Transform`` that doesn't use ``Objective``, instead it operates
    on lists of tensors directly.

    This has a ``concat_params`` setting which is used in quite a few modules, for example it is optional
    in all full-matrix method like Quasi-Newton or full-matrix Adagrad.

    To use, subclass this and override one of ``single_tensor_update`` or ``multi_tensor_update``,
    and one of ``single_tensor_apply`` or ``multi_tensor_apply``.
    """
    def __init__(
        self,
        defaults: dict[str, Any] | None = None,
        update_freq: int = 1,
        concat_params: bool = False,
        uses_grad: bool = False,
        uses_loss: bool = False,
        inner: Chainable | None = None,
    ):
        super().__init__(defaults, update_freq=update_freq, inner=inner)

        self._concat_params = concat_params
        self._uses_grad = uses_grad
        self._uses_loss = uses_loss

    def single_tensor_update(
        self,
        tensor: torch.Tensor,
        param: torch.Tensor,
        grad: torch.Tensor | None,
        loss: torch.Tensor | None,
        state: dict[str, Any],
        setting: Mapping[str, Any],
    ) -> None:
        """Updates ``state``. This should not modify ``tensor``."""

    def single_tensor_apply(
        self,
        tensor: torch.Tensor,
        param: torch.Tensor,
        grad: torch.Tensor | None,
        loss: torch.Tensor | None,
        state: dict[str, Any],
        setting: Mapping[str, Any],
    ) -> torch.Tensor:
        """Updates ``tensor`` and returns it. This shouldn't modify ``state`` if possible."""
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement `single_tensor_apply`.")

    def multi_tensor_update(
        self,
        tensors: list[torch.Tensor],
        params: list[torch.Tensor],
        grads: list[torch.Tensor] | None,
        loss: torch.Tensor | None,
        states: list[dict[str, Any]],
        settings: Sequence[Mapping[str, Any]],
    ) -> None:
        """Updates ``states``. This should not modify ``tensor``.
        By default calls ``single_tensor_update`` on all tensors."""

        if grads is None:
            grads = cast(list, [None] * len(tensors))

        for tensor, param, grad, state, setting in zip(tensors, params, grads, states, settings):
            self.single_tensor_update(tensor=tensor, param=param, grad=grad, loss=loss, state=state, setting=setting)

    def multi_tensor_apply(
        self,
        tensors: list[torch.Tensor],
        params: list[torch.Tensor],
        grads: list[torch.Tensor] | None,
        loss: torch.Tensor | None,
        states: list[dict[str, Any]],
        settings: Sequence[Mapping[str, Any]],
    ) -> list[torch.Tensor]:
        """Updates ``tensors`` and returns it. This shouldn't modify ``state`` if possible.
         By default calls ``single_tensor_apply`` on all tensors."""

        if grads is None:
            grads = cast(list, [None] * len(tensors))

        ret = []
        for tensor, param, grad, state, setting in zip(tensors, params, grads, states, settings):
            u = self.single_tensor_apply(tensor=tensor, param=param, grad=grad, loss=loss, state=state, setting=setting)
            ret.append(u)

        return ret

    def _get_grads_loss(self, objective: Objective):
        """evaluates grads and loss only if needed"""

        if self._uses_grad: grads = objective.get_grads()
        else: grads = None # better explicitly set to None rather than objective.grads because it shouldn't be used

        if self._uses_loss: loss = objective.get_loss(backward=False)
        else: loss = None

        return grads, loss

    def _get_cat_updates_params_grads(self, objective: Objective, grads: list[torch.Tensor] | None):
        assert self._concat_params

        cat_updates = [torch.cat([u.ravel() for u in objective.get_updates()])]
        cat_params = [torch.cat([p.ravel() for p in objective.params])]

        if grads is None: cat_grads = None
        else: cat_grads = [torch.cat([g.ravel() for g in grads])]

        return cat_updates, cat_params, cat_grads

    @final
    def update_states(self, objective: Objective, states: list[dict[str, Any]], settings: Sequence[Mapping[str, Any]]) -> None:
        """Updates ``states``. This should not modify ``objective.update``. Loss can be accessed by ``objective.get_loss()``."""

        grads, loss = self._get_grads_loss(objective)

        if self._concat_params:
            cat_updates, cat_params, cat_grads = self._get_cat_updates_params_grads(objective, grads)
            self.multi_tensor_update(
                tensors=cat_updates,
                params=cat_params,
                grads=cat_grads,
                loss=loss,
                states=[states[0]],
                settings=[settings[0]]
            )

        else:
            self.multi_tensor_update(
                tensors=objective.get_updates(),
                params=objective.params,
                grads=grads,
                loss=loss,
                states=states,
                settings=settings
            )

    @final
    def apply_states(self, objective: Objective, states: list[dict[str, Any]], settings: Sequence[Mapping[str, Any]]) -> Objective:
        """Updates ``objective`` using ``states`` and returns it."""
        # here objective might've been modified by `inner`
        # or within some functional logic
        # we have to re-cat
        grads, loss = self._get_grads_loss(objective)

        if self._concat_params:
            cat_updates, cat_params, cat_grads = self._get_cat_updates_params_grads(objective, grads)
            cat_updates = self.multi_tensor_apply(
                tensors=cat_updates,
                params=cat_params,
                grads=cat_grads,
                loss=loss,
                states=[states[0]],
                settings=[settings[0]]
            )
            objective.updates = vec_to_tensors(cat_updates[0], objective.params)

        else:
            objective.updates = self.multi_tensor_apply(
                tensors=objective.get_updates(),
                params=objective.params,
                grads=grads,
                loss=loss,
                states=states,
                settings=settings
            )

        return objective


    # make sure _concat_params, _uses_grad and _uses_loss are saved in `state_dict`
    def _extra_pack(self):
        return {
            "__concat_params": self._concat_params,
            "__uses_grad": self._uses_grad,
            "__uses_loss": self._uses_loss,
        }

    def _extra_unpack(self, d):
        self._concat_params = d["__concat_params"]
        self._uses_grad = d["__uses_grad"]
        self._uses_loss = d["__uses_loss"]
