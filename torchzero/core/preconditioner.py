from abc import ABC, abstractmethod
from collections import ChainMap, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, overload, final

import torch

from .module import Module, Chainable, Vars
from .transform import apply, Transform, Target
from ..utils import TensorList, vec_to_tensors

class Preconditioner(Transform):
    """Abstract class for a preconditioner."""
    def __init__(
        self,
        defaults: dict | None,
        uses_grad: bool,
        concat_params: bool = False,
        update_freq: int = 1,
        scale_first: bool = False,
        inner: Chainable | None = None,
        target: Target = "update",
    ):
        if defaults is None: defaults = {}
        defaults.update(dict(__update_freq=update_freq, __concat_params=concat_params, __scale_first=scale_first))
        super().__init__(defaults, uses_grad=uses_grad, target=target)

        if inner is not None:
            self.set_child('inner', inner)

    @abstractmethod
    def update(self, tensors: list[torch.Tensor], params:list[torch.Tensor], grads:list[torch.Tensor] | None, states: list[dict[str, Any]], settings: Sequence[Mapping[str, Any]]):
        """updates the preconditioner with `tensors`, any internal state should be stored using `keys`"""

    @abstractmethod
    def apply(self, tensors:list[torch.Tensor], params:list[torch.Tensor], grads:list[torch.Tensor] | None, states: list[dict[str, Any]], settings: Sequence[Mapping[str, Any]]) -> list[torch.Tensor]:
        """applies preconditioner to `tensors`, any internal state should be stored using `keys`"""

    def reset(self):
        """reset the internal state"""
        self.state.clear()
        self.global_state.clear()


    def _tensor_wise_transform(self, tensors:list[torch.Tensor], params:list[torch.Tensor], grads:list[torch.Tensor] | None, vars:Vars) -> list[torch.Tensor]:
        step = self.global_state.get('step', 0)
        states = [self.state[p] for p in params]
        settings = [self.settings[p] for p in params]
        global_settings = settings[0]
        update_freq = global_settings['__update_freq']

        scale_first = global_settings['__scale_first']
        scale_factor = 0
        if scale_first and step == 0:
            # initial step size guess from pytorch LBFGS
            scale_factor = TensorList(tensors).abs().sum()

        # update preconditioner
        if step % update_freq == 0:
            self.update(tensors=tensors, params=params, grads=grads, states=states, settings=settings)

        # step with inner
        if 'inner' in self.children:
            tensors = apply(self.children['inner'], tensors=tensors, params=params, grads=grads, vars=vars)

        # apply preconditioner
        tensors = self.apply(tensors=tensors, params=params, grads=grads, states=states, settings=settings)

        # scale initial step, when preconditioner might not have been applied
        if scale_first and step == 0:
            torch._foreach_div_(tensors, scale_factor)

        self.global_state['step'] = step + 1
        return tensors

    def _concat_transform(self, tensors:list[torch.Tensor], params:list[torch.Tensor], grads:list[torch.Tensor] | None, vars:Vars) -> list[torch.Tensor]:
        step = self.global_state.get('step', 0)
        tensors_vec = torch.cat([t.ravel() for t in tensors])
        params_vec = torch.cat([p.ravel() for p in params])
        grads_vec = [torch.cat([g.ravel() for g in grads])] if grads is not None else None

        states = [self.state[params[0]]]
        settings = [self.settings[params[0]]]
        global_settings = settings[0]
        update_freq = global_settings['__update_freq']

        scale_first = global_settings['__scale_first']
        scale_factor = 0
        if scale_first and step == 0:
            # initial step size guess from pytorch LBFGS
            scale_factor = tensors_vec.abs().sum()

        # update preconditioner
        if step % update_freq == 0:
            self.update(tensors=[tensors_vec], params=[params_vec], grads=grads_vec, states=states, settings=settings)

        # step with inner
        if 'inner' in self.children:
            tensors = apply(self.children['inner'], tensors=tensors, params=params, grads=grads, vars=vars)
            tensors_vec = torch.cat([t.ravel() for t in tensors]) # have to recat

        # apply preconditioner
        tensors_vec = self.apply(tensors=[tensors_vec], params=[params_vec], grads=grads_vec, states=states, settings=settings)[0]

        # scale initial step, when preconditioner might not have been applied
        if scale_first and step == 0:
            tensors_vec /= scale_factor

        tensors = vec_to_tensors(vec=tensors_vec, reference=tensors)
        self.global_state['step'] = step + 1
        return tensors

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        concat_params = self.settings[params[0]]['__concat_params']
        if concat_params: return self._concat_transform(tensors, params, grads, vars)
        return self._tensor_wise_transform(tensors, params, grads, vars)

class TensorwisePreconditioner(Preconditioner, ABC):
    @abstractmethod
    def update_tensor(self, tensor: torch.Tensor, param:torch.Tensor, grad: torch.Tensor | None, state: dict[str, Any], settings: Mapping[str, Any]):
        """update preconditioner with `tensor`"""

    @abstractmethod
    def apply_tensor(self, tensor: torch.Tensor, param:torch.Tensor, grad: torch.Tensor | None, state: dict[str, Any], settings: Mapping[str, Any]) -> torch.Tensor:
        """apply preconditioner to `tensor`"""

    @final
    def update(self, tensors, params, grads, states, settings):
        if grads is None: grads = [None]*len(tensors)
        for t,p,g,state,setting in zip(tensors, params, grads, states, settings):
            self.update_tensor(t, p, g, state, setting)

    @final
    def apply(self, tensors, params, grads, states, settings):
        preconditioned = []
        if grads is None: grads = [None]*len(tensors)
        for t,p,g,state,setting in zip(tensors, params, grads, states, settings):
            preconditioned.append(self.apply_tensor(t, p, g, state, setting))
        return preconditioned

