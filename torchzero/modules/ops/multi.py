#pyright: reportIncompatibleMethodOverride=false
""""""
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from operator import itemgetter
from typing import Any, Literal

import torch

from ...core import Chainable, Module, Target, Var, maybe_chain
from ...utils import TensorList, tensorlist


class MultiOperationBase(Module, ABC):
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



class SubModules(MultiOperationBase):
    """Calculates :code:`input - other`. :code:`input` and :code:`other` can be numbers or modules."""
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

class DivModules(MultiOperationBase):
    """Calculates :code:`input / other`. :code:`input` and :code:`other` can be numbers or modules."""
    def __init__(self, input: Chainable | float, other: Chainable | float, other_first:bool=False):
        defaults = {}
        if other_first: super().__init__(defaults, other=other, input=input)
        else: super().__init__(defaults, input=input, other=other)

    @torch.no_grad
    def transform(self, var: Var, input: float | list[torch.Tensor], other: float | list[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(input, (int,float)):
            assert isinstance(other, list)
            return input / TensorList(other)

        torch._foreach_div_(input, other)
        return input


class PowModules(MultiOperationBase):
    """Calculates :code:`input ** exponent`. :code:`input` and :code:`other` can be numbers or modules."""
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

class LerpModules(MultiOperationBase):
    """Does a linear interpolation of :code:`input(tensors)` and :code:`end(tensors)` based on a scalar :code:`weight`.

    The output is given by :code:`output = input(tensors) + weight * (end(tensors) - input(tensors))`
    """
    def __init__(self, input: Chainable, end: Chainable, weight: float):
        defaults = dict(weight=weight)
        super().__init__(defaults, input=input, end=end)

    @torch.no_grad
    def transform(self, var: Var, input: list[torch.Tensor], end: list[torch.Tensor]) -> list[torch.Tensor]:
        torch._foreach_lerp_(input, end, weight=self.settings[var.params[0]]['weight'])
        return input

class ClipModules(MultiOperationBase):
    """Calculates :code:`input(tensors).clip(min, max)`. :code:`min` and :code:`max` can be numbers or modules."""
    def __init__(self, input: Chainable, min: float | Chainable | None = None, max: float | Chainable | None = None):
        defaults = {}
        super().__init__(defaults, input=input, min=min, max=max)

    @torch.no_grad
    def transform(self, var: Var, input: list[torch.Tensor], min: float | list[torch.Tensor], max: float | list[torch.Tensor]) -> list[torch.Tensor]:
        return TensorList(input).clamp_(min=min, max=max)


class GraftModules(MultiOperationBase):
    """Outputs :code:`direction` output rescaled to have the same norm as :code:`magnitude` output.

    Args:
        direction (Chainable): module to use the direction from
        magnitude (Chainable): module to use the magnitude from
        tensorwise (bool, optional): whether to calculate norm per-tensor or globally. Defaults to True.
        ord (float, optional): norm order. Defaults to 2.
        eps (float, optional): clips denominator to be no less than this value. Defaults to 1e-6.
        strength (float, optional): strength of grafting. Defaults to 1.

    Example:
        Shampoo grafted to Adam

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.GraftModules(
                    direction = tz.m.Shampoo(),
                    magnitude = tz.m.Adam(),
                ),
                tz.m.LR(1e-3)
            )

    Reference:
        Agarwal, N., Anil, R., Hazan, E., Koren, T., & Zhang, C. (2020). Disentangling adaptive gradient methods from learning rates. arXiv preprint arXiv:2002.11803. https://arxiv.org/pdf/2002.11803
    """
    def __init__(self, direction: Chainable, magnitude: Chainable, tensorwise:bool=True, ord:float=2, eps:float = 1e-6, strength:float=1):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps, strength=strength)
        super().__init__(defaults, direction=direction, magnitude=magnitude)

    @torch.no_grad
    def transform(self, var, magnitude: list[torch.Tensor], direction:list[torch.Tensor]):
        tensorwise, ord, eps, strength = itemgetter('tensorwise','ord','eps', 'strength')(self.settings[var.params[0]])
        return TensorList(direction).graft_(magnitude, tensorwise=tensorwise, ord=ord, eps=eps, strength=strength)

class MultiplyByModuleNorm(MultiOperationBase):
    """Outputs :code:`input` multiplied by norm of the :code:`norm` output."""
    def __init__(self, input: Chainable, norm: Chainable, tensorwise:bool=True, ord:float|Literal['mean_abs']=2):
        defaults = dict(tensorwise=tensorwise, ord=ord)
        super().__init__(defaults, input=input, norm=norm)

    @torch.no_grad
    def transform(self, var, input: list[torch.Tensor], norm:list[torch.Tensor]):
        tensorwise, ord = itemgetter('tensorwise','ord')(self.settings[var.params[0]])
        if tensorwise:
            if ord == 'mean_abs': n = [t.mean() for t in torch._foreach_abs(norm)]
            else: n = torch._foreach_norm(norm, ord)
        else: n = TensorList(norm).global_vector_norm(ord)

        torch._foreach_mul_(input, n)
        return input

class DivideByModuleNorm(MultiOperationBase):
    """Outputs :code:`input` divided by norm of the :code:`norm` output."""
    def __init__(self, input: Chainable, norm: Chainable, tensorwise:bool=True, ord:float|Literal['mean_abs']=2):
        defaults = dict(tensorwise=tensorwise, ord=ord)
        super().__init__(defaults, input=input, norm=norm)

    @torch.no_grad
    def transform(self, var, input: list[torch.Tensor], norm:list[torch.Tensor]):
        tensorwise, ord = itemgetter('tensorwise','ord')(self.settings[var.params[0]])
        if tensorwise:
            if ord == 'mean_abs': n = [t.mean().clip(min=1e-8) for t in torch._foreach_abs(norm)]
            else: n = torch._foreach_clamp_min(torch._foreach_norm(norm, ord), 1e-8)
        else: n = TensorList(norm).global_vector_norm(ord).clip(min=1e-8)

        torch._foreach_div_(input, n)
        return input