""""""
from abc import ABC, abstractmethod
from collections.abc import Iterable,Sequence
from typing import Any, cast

import torch

from ...core import Chainable, Module, Target, Var, maybe_chain


class ReduceOperationBase(Module, ABC):
    """Base class for reduction operations like Sum, Prod, Maximum. This is an abstract class, subclass it and override `transform` method to use it."""
    def __init__(self, defaults: dict[str, Any] | None, *operands: Chainable | Any):
        super().__init__(defaults=defaults)

        self.operands = []
        for i, v in enumerate(operands):

            if isinstance(v, (Module, Sequence)):
                self.set_child(f'operand_{i}', v)
                self.operands.append(self.children[f'operand_{i}'])
            else:
                self.operands.append(v)

        if not self.children:
            raise ValueError('At least one operand must be a module')

    @abstractmethod
    def transform(self, var: Var, *operands: Any | list[torch.Tensor]) -> list[torch.Tensor]:
        """applies the operation to operands"""
        raise NotImplementedError

    @torch.no_grad
    def step(self, var: Var) -> Var:
        # pass cloned update to all module operands
        processed_operands: list[Any | list[torch.Tensor]] = self.operands.copy()

        for i, v in enumerate(self.operands):
            if f'operand_{i}' in self.children:
                v: Module
                updated_var = v.step(var.clone(clone_update=True))
                processed_operands[i] = updated_var.get_update()
                var.update_attrs_from_clone_(updated_var) # update loss, grad, etc if this module calculated them

        transformed = self.transform(var, *processed_operands)
        var.update = transformed
        return var

class Sum(ReduceOperationBase):
    """Outputs sum of :code:`inputs` that can be modules or numbers."""
    USE_MEAN = False
    def __init__(self, *inputs: Chainable | float):
        super().__init__({}, *inputs)

    @torch.no_grad
    def transform(self, var: Var, *inputs: float | list[torch.Tensor]) -> list[torch.Tensor]:
        sorted_inputs = sorted(inputs, key=lambda x: isinstance(x, float))
        sum = cast(list, sorted_inputs[0])
        if len(sorted_inputs) > 1:
            for v in sorted_inputs[1:]:
                torch._foreach_add_(sum, v)

        if self.USE_MEAN and len(sorted_inputs) > 1: torch._foreach_div_(sum, len(sorted_inputs))
        return sum

class Mean(Sum):
    """Outputs a mean of :code:`inputs` that can be modules or numbers."""
    USE_MEAN = True


class WeightedSum(ReduceOperationBase):
    USE_MEAN = False
    def __init__(self, *inputs: Chainable | float, weights: Iterable[float]):
        """Outputs a weighted sum of :code:`inputs` that can be modules or numbers."""
        weights = list(weights)
        if len(inputs) != len(weights):
            raise ValueError(f'Number of inputs {len(inputs)} must match number of weights {len(weights)}')
        defaults = dict(weights=weights)
        super().__init__(defaults=defaults, *inputs)

    @torch.no_grad
    def transform(self, var: Var, *inputs: float | list[torch.Tensor]) -> list[torch.Tensor]:
        sorted_inputs = sorted(inputs, key=lambda x: isinstance(x, float))
        weights = self.settings[var.params[0]]['weights']
        sum = cast(list, sorted_inputs[0])
        torch._foreach_mul_(sum, weights[0])
        if len(sorted_inputs) > 1:
            for v, w in zip(sorted_inputs[1:], weights[1:]):
                if isinstance(v, (int, float)): torch._foreach_add_(sum, v*w)
                else: torch._foreach_add_(sum, v, alpha=w)

        if self.USE_MEAN and len(sorted_inputs) > 1: torch._foreach_div_(sum, len(sorted_inputs))
        return sum


class WeightedMean(WeightedSum):
    """Outputs weighted mean of :code:`inputs` that can be modules or numbers."""
    USE_MEAN = True

class Median(ReduceOperationBase):
    """Outputs median of :code:`inputs` that can be modules or numbers."""
    def __init__(self, *inputs: Chainable | float):
        super().__init__({}, *inputs)

    @torch.no_grad
    def transform(self, var: Var, *inputs: float | list[torch.Tensor]) -> list[torch.Tensor]:
        res = []
        lists = [i for i in inputs if isinstance(i, list)]
        floats = [i for i in inputs if isinstance(i, (int,float))]
        for tensors in zip(*lists):
            res.append(torch.median(torch.stack(tensors + tuple(torch.full_like(tensors[0], f) for f in floats)), dim=0))
        return res

class Prod(ReduceOperationBase):
    """Outputs product of :code:`inputs` that can be modules or numbers."""
    def __init__(self, *inputs: Chainable | float):
        super().__init__({}, *inputs)

    @torch.no_grad
    def transform(self, var: Var, *inputs: float | list[torch.Tensor]) -> list[torch.Tensor]:
        sorted_inputs = sorted(inputs, key=lambda x: isinstance(x, float))
        prod = cast(list, sorted_inputs[0])
        if len(sorted_inputs) > 1:
            for v in sorted_inputs[1:]:
                torch._foreach_mul_(prod, v)

        return prod

class MaximumModules(ReduceOperationBase):
    """Outputs elementwise maximum of :code:`inputs` that can be modules or numbers."""
    def __init__(self, *inputs: Chainable | float):
        super().__init__({}, *inputs)

    @torch.no_grad
    def transform(self, var: Var, *inputs: float | list[torch.Tensor]) -> list[torch.Tensor]:
        sorted_inputs = sorted(inputs, key=lambda x: isinstance(x, float))
        maximum = cast(list, sorted_inputs[0])
        if len(sorted_inputs) > 1:
            for v in sorted_inputs[1:]:
                torch._foreach_maximum_(maximum, v)

        return maximum

class MinimumModules(ReduceOperationBase):
    """Outputs elementwise minimum of :code:`inputs` that can be modules or numbers."""
    def __init__(self, *inputs: Chainable | float):
        super().__init__({}, *inputs)

    @torch.no_grad
    def transform(self, var: Var, *inputs: float | list[torch.Tensor]) -> list[torch.Tensor]:
        sorted_inputs = sorted(inputs, key=lambda x: isinstance(x, float))
        minimum = cast(list, sorted_inputs[0])
        if len(sorted_inputs) > 1:
            for v in sorted_inputs[1:]:
                torch._foreach_minimum_(minimum, v)

        return minimum
