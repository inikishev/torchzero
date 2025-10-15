from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from operator import itemgetter
from typing import TYPE_CHECKING, Any, cast, final

import torch

from ..utils import safe_dict_update_, vec_to_tensors
from .module import Module

if TYPE_CHECKING:
    from .chain import Chainable
    from .objective import Objective


class Transform(Module):
    """``Transform`` is a ``Module`` with only optional children.

    ``Transform`` if more flexible in that as long as there are no children, it can use a custom list of states
    and settings instead of ``self.state`` and ``self.setting``.

    To use, subclass this and override ``update_states`` and ``apply_states``.
    """
    def __init__(self, defaults: dict[str, Any] | None = None, update_freq: int = 1, inner: "Chainable | None" = None):

        # store update_freq in defaults so that it is scheduleable
        if defaults is None: defaults = {}
        safe_dict_update_(defaults, {"__update_freq": update_freq})

        super().__init__(defaults)

        self._objective = None
        if inner is not None:
            self.set_child("__inner", inner)

    # settings shouldn't mutate, so they are typed as Sequence[Mapping]
    def update_states(self, objective: "Objective", states: list[dict[str, Any]], settings: Sequence[Mapping[str, Any]]) -> None:
        """Updates ``states``. This should not modify ``objective.update``."""

    @abstractmethod
    def apply_states(self, objective: "Objective", states: list[dict[str, Any]], settings: Sequence[Mapping[str, Any]]) -> "Objective":
        """Updates ``objective`` using ``states``."""

    def _get_states_settings(self, objective: "Objective") -> tuple[list, tuple]:
        # itemgetter is faster
        # but need to make sure it returns a tuple, as if there is a single param, it returns the value
        getter = itemgetter(*objective.params)
        is_single = len(objective.params) == 1
        states = getter(self.state)
        settings = getter(self.settings)

        if is_single:
            states = [states, ]
            settings = (settings, )

        else:
            states = list(states) # itemgetter returns tuple

        return states, settings

    @final
    def update(self, objective:"Objective"):
        step = self.increment_counter("__step", 0)

        if step % self.settings[objective.params[0]]["__update_freq"] == 0:
            states, settings = self._get_states_settings(objective)
            self.update_states(objective=objective, states=states, settings=settings)

    @final
    def apply(self, objective: "Objective"):

        # inner step
        if "__inner" in self.children:
            inner = self.children["__inner"]
            objective = inner.step(objective)

        # apply and return
        states, settings = self._get_states_settings(objective)
        return self.apply_states(objective=objective, states=states, settings=settings)



class TensorTransform(Transform):
    """``TensorTransform`` is a ``Transform`` that doesn't use ``Objective``, instead it operates
    on lists of tensors directly.

    This has a ``concat_params`` setting which is used in quite a few modules, for example it is optional
    in all full-matrix method like Quasi-Newton or full-matrix Adagrad.

    To use, subclass this and override one of ``single_tensor_update`` or ``multi_tensor_update``,
    and one of ``single_tensor_apply`` or ``multi_tensor_apply``.

    For copying:

    multi tensor:
    ```
    def multi_tensor_initialize(self, tensors, params, grads, loss, states, settings):
        ...
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
        ...
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        ...
    ```

    single tensor:

    ```
    def single_tensor_initialize(self, tensor, param, grad, loss, state, setting):
        ...
    def single_tensor_update(self, tensor, param, grad, loss, state, setting):
        ...
    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):
        ...
    ```
    """
    def __init__(
        self,
        defaults: dict[str, Any] | None = None,
        update_freq: int = 1,
        concat_params: bool = False,
        uses_grad: bool = False,
        uses_loss: bool = False,
        inner: "Chainable | None" = None,
    ):
        super().__init__(defaults, update_freq=update_freq, inner=inner)

        self._concat_params = concat_params
        self._uses_grad = uses_grad
        self._uses_loss = uses_loss


    # ------------------------------- single tensor ------------------------------ #
    def single_tensor_initialize(
        self,
        tensor: torch.Tensor,
        param: torch.Tensor,
        grad: torch.Tensor | None,
        loss: torch.Tensor | None,
        state: dict[str, Any],
        setting: Mapping[str, Any],
    ) -> None:
        """initialize ``state`` before first ``update``.
        """

    def single_tensor_update(
        self,
        tensor: torch.Tensor,
        param: torch.Tensor,
        grad: torch.Tensor | None,
        loss: torch.Tensor | None,
        state: dict[str, Any],
        setting: Mapping[str, Any],
    ) -> None:
        """Updates ``state``. This should not modify ``tensor``.
        """

    def single_tensor_apply(
        self,
        tensor: torch.Tensor,
        param: torch.Tensor,
        grad: torch.Tensor | None,
        loss: torch.Tensor | None,
        state: dict[str, Any],
        setting: Mapping[str, Any],
    ) -> torch.Tensor:
        """Updates ``tensor`` and returns it. This shouldn't modify ``state`` if possible.
        """
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement `single_tensor_apply`.")

    # ------------------------------- multi tensor ------------------------------- #
    def multi_tensor_initialize(
        self,
        tensors: list[torch.Tensor],
        params: list[torch.Tensor],
        grads: list[torch.Tensor] | None,
        loss: torch.Tensor | None,
        states: list[dict[str, Any]],
        settings: Sequence[Mapping[str, Any]],
    ) -> None:
        """initialize ``states`` before first ``update``.
        By default calls ``single_tensor_initialize`` on all tensors.
        """
        if grads is None:
            grads = cast(list, [None] * len(tensors))

        for tensor, param, grad, state, setting in zip(tensors, params, grads, states, settings):
            self.single_tensor_initialize(tensor=tensor, param=param, grad=grad, loss=loss, state=state, setting=setting)

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
        By default calls ``single_tensor_update`` on all tensors.
        """

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
    ) -> Sequence[torch.Tensor]:
        """Updates ``tensors`` and returns it. This shouldn't modify ``state`` if possible.
         By default calls ``single_tensor_apply`` on all tensors.
         """

        if grads is None:
            grads = cast(list, [None] * len(tensors))

        ret = []
        for tensor, param, grad, state, setting in zip(tensors, params, grads, states, settings):
            u = self.single_tensor_apply(tensor=tensor, param=param, grad=grad, loss=loss, state=state, setting=setting)
            ret.append(u)

        return ret

    def _get_grads_loss(self, objective: "Objective"):
        """evaluates grads and loss only if needed"""

        if self._uses_grad: grads = objective.get_grads()
        else: grads = None # better explicitly set to None rather than objective.grads because it shouldn't be used

        if self._uses_loss: loss = objective.get_loss(backward=True)
        else: loss = None

        return grads, loss

    @torch.no_grad
    def _get_cat_updates_params_grads(self, objective: "Objective", grads: list[torch.Tensor] | None):
        assert self._concat_params

        cat_updates = [torch.cat([u.ravel() for u in objective.get_updates()])]
        cat_params = [torch.cat([p.ravel() for p in objective.params])]

        if grads is None: cat_grads = None
        else: cat_grads = [torch.cat([g.ravel() for g in grads])]

        return cat_updates, cat_params, cat_grads

    def _gather_tensors(self, objective: "Objective", states: list[dict[str, Any]], settings: Sequence[Mapping[str, Any]]):
        """returns everything for ``multi_tensor_*``. Concatenates if ```self._concat_params``.
        evaluates grads and loss if ``self._uses_grad`` and ``self._uses_loss``"""

        # evaluate grads and loss if `self._uses_grad` and `self._uses_loss`
        grads, loss = self._get_grads_loss(objective)

        # gather all things
        # concatenate everything to a vec if `self._concat_params`
        if self._concat_params:
            tensors, params, grads = self._get_cat_updates_params_grads(objective, grads)
            states = [states[0]]; settings = [settings[0]]

        # or take original values
        else:
            tensors=objective.get_updates()
            params = objective.params

        return tensors, params, grads, loss, states, settings

    @final
    def update_states(self, objective: "Objective", states: list[dict[str, Any]], settings: Sequence[Mapping[str, Any]]) -> None:
        tensors, params, grads, loss, states, settings = self._gather_tensors(objective, states, settings)

        # initialize before the first update
        num_updates = self.increment_counter("__num_updates", 0)
        if num_updates == 0:
            self.multi_tensor_initialize(
                tensors=tensors,
                params=params,
                grads=grads,
                loss=loss,
                states=states,
                settings=settings
            )

        # update
        self.multi_tensor_update(
            tensors=tensors,
            params=params,
            grads=grads,
            loss=loss,
            states=states,
            settings=settings
        )

    @final
    def apply_states(self, objective: "Objective", states: list[dict[str, Any]], settings: Sequence[Mapping[str, Any]]) -> "Objective":
        tensors, params, grads, loss, states, settings = self._gather_tensors(objective, states, settings)
        # note: _gather tensors will re-cat again if `_concat_params`, this is necessary because objective
        # may have been modified in functional logic, there is no way to know if that happened

        # apply
        ret = self.multi_tensor_apply(
            tensors=tensors,
            params=params,
            grads=grads,
            loss=loss,
            states=states,
            settings=settings
        )

        # uncat if needed and set objective.updates and return objective
        if self._concat_params:
            objective.updates = vec_to_tensors(ret[0], objective.params)

        else:
            objective.updates = list(ret)

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
