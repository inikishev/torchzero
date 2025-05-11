from abc import ABC, abstractmethod
from collections import ChainMap, defaultdict
from typing import Any, overload, final

import torch

from .module import Module, Chainable, Vars
from .transform import apply, Transform, Target
from ..utils import TensorList, vec_to_tensors

class Preconditioner(ABC):
    def __init__(self):
        self.state: dict[Any, dict[str, Any]] = defaultdict(dict)
        self.global_state: dict[Any,Any] = {}

    @abstractmethod
    def update(self, tensors: list[torch.Tensor], params:list[torch.Tensor], grads:list[torch.Tensor] | None, keys: list[Any]):
        """updates the preconditioner with `tensors`, any internal state should be stored using `keys`"""

    @abstractmethod
    def apply(self, tensors:list[torch.Tensor], params:list[torch.Tensor], grads:list[torch.Tensor] | None, keys: list[Any]) -> list[torch.Tensor]:
        """applies preconditioner to `tensors`, any internal state should be stored using `keys`"""

    def reset(self):
        """reset the internal state"""
        self.state.clear()
        self.global_state.clear()


class TensorwisePreconditioner(Preconditioner, ABC):
    @abstractmethod
    def update_tensor(self, tensor: torch.Tensor, param:torch.Tensor, grad: torch.Tensor | None, state: dict[str, Any]):
        """update preconditioner with `tensor`"""

    @abstractmethod
    def apply_tensor(self, tensor: torch.Tensor, param:torch.Tensor, grad: torch.Tensor | None, state: dict[str, Any]) -> torch.Tensor:
        """apply preconditioner to `tensor`"""

    @final
    def update(self, tensors, params, grads, keys):
        if grads is None: grads = [None]*len(tensors)
        for t,p,g,k in zip(tensors, params, grads, keys): # we assume tensors is grads but anything can be passed
            self.update_tensor(t, p, g, self.state[k])

    @final
    def apply(self, tensors, params, grads, keys):
        preconditioned = []
        if grads is None: grads = [None]*len(tensors)
        for t,p,g,k in zip(tensors, params, grads, keys): # we assume tensors is grads but anything can be passed
            preconditioned.append(self.apply_tensor(t, p, g, self.state[k]))
        return preconditioned


class Precondition(Transform):
    def __init__(
        self,
        preconditioner: Preconditioner,
        uses_grad: bool,
        tensorwise: bool = True,
        update_freq: int = 1,
        scale_first: bool = False,
        inner: Chainable | None = None,
        target: Target = "update",
    ):
        defaults = dict(update_freq=update_freq, tensorwise=tensorwise, scale_first=scale_first)
        super().__init__(defaults, uses_grad=uses_grad, target=target)
        self.preconditioner = preconditioner

        if inner is not None:
            self.set_child('inner', inner)

    def _tensor_wise_transform(self, tensors:list[torch.Tensor], params:list[torch.Tensor], grads:list[torch.Tensor] | None, vars:Vars) -> list[torch.Tensor]:
        step = self.global_state.get('step', 0)
        settings = self.settings[params[0]]
        update_freq = settings['update_freq']

        scale_first = settings['scale_first']
        scale_factor = 0
        if scale_first and step == 0:
            # initial step size guess from pytorch LBFGS
            scale_factor = TensorList(tensors).abs().sum()

        # update preconditioner
        if step % update_freq == 0:
            self.preconditioner.update(tensors=tensors, params=params, grads=grads, keys=params)

        # step with inner
        if 'inner' in self.children:
            tensors = apply(self.children['inner'], tensors=tensors, params=params, grads=grads, vars=vars)

        # apply preconditioner
        tensors = self.preconditioner.apply(tensors=tensors, params=params, grads=grads, keys=params)

        # scale initial step, when preconditioner might not have been applied
        if scale_first and step == 0:
            torch._foreach_div_(tensors, scale_factor)

        self.global_state['step'] = step + 1
        return tensors

    def _vec_transform(self, tensors:list[torch.Tensor], params:list[torch.Tensor], grads:list[torch.Tensor] | None, vars:Vars) -> list[torch.Tensor]:
        step = self.global_state.get('step', 0)
        tensors_vec = torch.cat([t.ravel() for t in tensors])
        params_vec = torch.cat([p.ravel() for p in params])
        grads_vec = [torch.cat([g.ravel() for g in grads])] if grads is not None else None

        settings = self.settings[params[0]]
        update_freq = settings['update_freq']

        scale_first = settings['scale_first']
        scale_factor = 0
        if scale_first and step == 0:
            # initial step size guess from pytorch LBFGS
            scale_factor = tensors_vec.abs().sum()

        # update preconditioner
        if step % update_freq == 0:
            self.preconditioner.update(tensors=[tensors_vec], params=[params_vec], grads=grads_vec, keys=['vec'])

        # step with inner
        if 'inner' in self.children:
            tensors = apply(self.children['inner'], tensors=tensors, params=params, grads=grads, vars=vars)
            tensors_vec = torch.cat([t.ravel() for t in tensors]) # have to recat

        # apply preconditioner
        tensors_vec = self.preconditioner.apply(tensors=[tensors_vec], params=[params_vec], grads=grads_vec, keys=['vec'])[0]

        # scale initial step, when preconditioner might not have been applied
        if scale_first and step == 0:
            tensors_vec /= scale_factor

        tensors = vec_to_tensors(vec=tensors_vec, reference=tensors)
        self.global_state['step'] = step + 1
        return tensors

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        tensorwise = self.settings[params[0]]['tensorwise']
        if tensorwise: return self._tensor_wise_transform(tensors, params, grads, vars)
        return self._vec_transform(tensors, params, grads, vars)