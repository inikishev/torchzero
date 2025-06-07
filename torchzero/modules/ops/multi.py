#pyright: reportIncompatibleMethodOverride=false
""""""
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from operator import itemgetter
from typing import Any

import torch

from ...core import Chainable, Module, Target, Var, maybe_chain
from ...utils import TensorList, tensorlist


class MultiOperation(Module, ABC):
    """Base class for operations that use operands. This is an abstract class, subclass it and override `transform` method to use it."""
    def __init__(self, defaults: dict[str, Any] | None, **operands: Chainable | Any):
        super().__init__(defaults=defaults)

        self.operands = {}
        for k,v in operands.items():

            if isinstance(v, (Module, Sequence)):
                self.set_child(k, v)
                self.operands[k] = self.children[k]
            else:
                self.operands[k] = v

        if not self.children:
            raise ValueError('At least one operand must be a module')

    @abstractmethod
    def transform(self, var: Var, **operands: Any | list[torch.Tensor]) -> list[torch.Tensor]:
        """applies the operation to operands"""
        raise NotImplementedError

    @torch.no_grad
    def step(self, var: Var) -> Var:
        # pass cloned update to all module operands
        processed_operands: dict[str, Any | list[torch.Tensor]] = self.operands.copy()

        for k,v in self.operands.items():
            if k in self.children:
                v: Module
                updated_var = v.step(var.clone(clone_update=True))
                processed_operands[k] = updated_var.get_update()
                var.update_attrs_from_clone_(updated_var) # update loss, grad, etc if this module calculated them

        transformed = self.transform(var, **processed_operands)
        var.update = transformed
        return var



class SubModules(MultiOperation):
    def __init__(self, input: Chainable | float, other: Chainable | float, alpha: float = 1):
        defaults = dict(alpha=alpha)
        super().__init__(defaults, input=input, other=other)

    @torch.no_grad
    def transform(self, var: Var, input: float | list[torch.Tensor], other: float | list[torch.Tensor]) -> list[torch.Tensor]:
        alpha = self.settings[var.params[0]]['alpha']

        if isinstance(input, (int,float)):
            assert isinstance(other, list)
            return input - TensorList(other).mul_(alpha)

        if isinstance(other, (int, float)): torch._foreach_sub_(input, other * alpha)
        else: torch._foreach_sub_(input, other, alpha=alpha)
        return input

class DivModules(MultiOperation):
    def __init__(self, input: Chainable | float, other: Chainable | float):
        defaults = {}
        super().__init__(defaults, input=input, other=other)

    @torch.no_grad
    def transform(self, var: Var, input: float | list[torch.Tensor], other: float | list[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(input, (int,float)):
            assert isinstance(other, list)
            return input / TensorList(other)

        torch._foreach_div_(input, other)
        return input

class PowModules(MultiOperation):
    def __init__(self, input: Chainable | float, exponent: Chainable | float):
        defaults = {}
        super().__init__(defaults, input=input, exponent=exponent)

    @torch.no_grad
    def transform(self, var: Var, input: float | list[torch.Tensor], exponent: float | list[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(input, (int,float)):
            assert isinstance(exponent, list)
            return input ** TensorList(exponent)

        torch._foreach_div_(input, exponent)
        return input

class LerpModules(MultiOperation):
    def __init__(self, input: Chainable, end: Chainable, weight: float):
        defaults = dict(weight=weight)
        super().__init__(defaults, input=input, end=end)

    @torch.no_grad
    def transform(self, var: Var, input: list[torch.Tensor], end: list[torch.Tensor]) -> list[torch.Tensor]:
        torch._foreach_lerp_(input, end, weight=self.settings[var.params[0]]['weight'])
        return input

class ClipModules(MultiOperation):
    def __init__(self, input: Chainable, min: float | Chainable | None = None, max: float | Chainable | None = None):
        defaults = {}
        super().__init__(defaults, input=input, min=min, max=max)

    @torch.no_grad
    def transform(self, var: Var, input: list[torch.Tensor], min: float | list[torch.Tensor], max: float | list[torch.Tensor]) -> list[torch.Tensor]:
        return TensorList(input).clamp_(min=min, max=max)


class GraftModules(MultiOperation):
    def __init__(self, direction: Chainable, magnitude: Chainable, tensorwise:bool=True, ord:float=2, eps:float = 1e-6, strength:float=1):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps, strength=strength)
        super().__init__(defaults, direction=direction, magnitude=magnitude)

    @torch.no_grad
    def transform(self, var, magnitude: list[torch.Tensor], direction:list[torch.Tensor]):
        tensorwise, ord, eps, strength = itemgetter('tensorwise','ord','eps', 'strength')(self.settings[var.params[0]])
        return TensorList(direction).graft_(magnitude, tensorwise=tensorwise, ord=ord, eps=eps, strength=strength)


class Where(MultiOperation):
    def __init__(self, condition: Chainable, input: Chainable | float, other: Chainable | float):
        super().__init__({}, condition=condition, input=input, other=other)

    @torch.no_grad
    def transform(self, var, condition: list[torch.Tensor], input: list[torch.Tensor] | float, other: list[torch.Tensor] | float):
        return tensorlist.where(TensorList(condition).as_bool(), input, other)

