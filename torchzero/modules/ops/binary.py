#pyright: reportIncompatibleMethodOverride=false
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from operator import itemgetter
from typing import Any

import torch

from ...core import Chainable, Module, Objective
from ...utils import TensorList


class BinaryOperationBase(Module, ABC):
    """Base class for operations that use update as the first operand. This is an abstract class, subclass it and override `transform` method to use it."""
    def __init__(self, defaults: dict[str, Any] | None, **operands: Chainable | Any):
        super().__init__(defaults=defaults)

        self.operands = {}
        for k,v in operands.items():

            if isinstance(v, (Module, Sequence)):
                self.set_child(k, v)
                self.operands[k] = self.children[k]
            else:
                self.operands[k] = v

    @abstractmethod
    def transform(self, objective: Objective, update: list[torch.Tensor], **operands: Any | list[torch.Tensor]) -> Iterable[torch.Tensor]:
        """applies the operation to operands"""
        raise NotImplementedError

    def update(self, objective): raise RuntimeError
    def apply(self, objective): raise RuntimeError

    @torch.no_grad
    def step(self, objective: Objective) -> Objective:
        # pass cloned update to all module operands
        processed_operands: dict[str, Any | list[torch.Tensor]] = self.operands.copy()

        for k,v in self.operands.items():
            if k in self.children:
                v: Module
                updated_obj = v.step(objective.clone(clone_updates=True))
                processed_operands[k] = updated_obj.get_updates()
                objective.update_attrs_from_clone_(updated_obj) # update loss, grad, etc if this module calculated them

        transformed = self.transform(objective, update=objective.get_updates(), **processed_operands)
        objective.updates = list(transformed)
        return objective


class Add(BinaryOperationBase):
    """Add ``other`` to tensors. ``other`` can be a number or a module.

    If ``other`` is a module, this calculates ``tensors + other(tensors)``
    """
    def __init__(self, other: Chainable | float, alpha: float = 1):
        defaults = dict(alpha=alpha)
        super().__init__(defaults, other=other)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], other: float | list[torch.Tensor]):
        if isinstance(other, (int,float)): torch._foreach_add_(update, other * self.defaults['alpha'])
        else: torch._foreach_add_(update, other, alpha=self.defaults['alpha'])
        return update

class Sub(BinaryOperationBase):
    """Subtract ``other`` from tensors. ``other`` can be a number or a module.

    If ``other`` is a module, this calculates :code:`tensors - other(tensors)`
    """
    def __init__(self, other: Chainable | float, alpha: float = 1):
        defaults = dict(alpha=alpha)
        super().__init__(defaults, other=other)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], other: float | list[torch.Tensor]):
        if isinstance(other, (int,float)): torch._foreach_sub_(update, other * self.defaults['alpha'])
        else: torch._foreach_sub_(update, other, alpha=self.defaults['alpha'])
        return update

class RSub(BinaryOperationBase):
    """Subtract tensors from ``other``. ``other`` can be a number or a module.

    If ``other`` is a module, this calculates ``other(tensors) - tensors``
    """
    def __init__(self, other: Chainable | float):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], other: float | list[torch.Tensor]):
        return other - TensorList(update)

class Mul(BinaryOperationBase):
    """Multiply tensors by ``other``. ``other`` can be a number or a module.

    If ``other`` is a module, this calculates ``tensors * other(tensors)``
    """
    def __init__(self, other: Chainable | float):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], other: float | list[torch.Tensor]):
        torch._foreach_mul_(update, other)
        return update

class Div(BinaryOperationBase):
    """Divide tensors by ``other``. ``other`` can be a number or a module.

    If ``other`` is a module, this calculates ``tensors / other(tensors)``
    """
    def __init__(self, other: Chainable | float):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], other: float | list[torch.Tensor]):
        torch._foreach_div_(update, other)
        return update

class RDiv(BinaryOperationBase):
    """Divide ``other`` by tensors. ``other`` can be a number or a module.

    If ``other`` is a module, this calculates ``other(tensors) / tensors``
    """
    def __init__(self, other: Chainable | float):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], other: float | list[torch.Tensor]):
        return other / TensorList(update)

class Pow(BinaryOperationBase):
    """Take tensors to the power of ``exponent``. ``exponent`` can be a number or a module.

    If ``exponent`` is a module, this calculates ``tensors ^ exponent(tensors)``
    """
    def __init__(self, exponent: Chainable | float):
        super().__init__({}, exponent=exponent)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], exponent: float | list[torch.Tensor]):
        torch._foreach_pow_(update, exponent)
        return update

class RPow(BinaryOperationBase):
    """Take ``other`` to the power of tensors. ``other`` can be a number or a module.

    If ``other`` is a module, this calculates ``other(tensors) ^ tensors``
    """
    def __init__(self, other: Chainable | float):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], other: float | list[torch.Tensor]):
        if isinstance(other, (int, float)): return torch._foreach_pow(other, update) # no in-place
        torch._foreach_pow_(other, update)
        return other

class Lerp(BinaryOperationBase):
    """Does a linear interpolation of tensors and ``end`` module based on a scalar ``weight``.

    The output is given by ``output = tensors + weight * (end(tensors) - tensors)``
    """
    def __init__(self, end: Chainable, weight: float):
        defaults = dict(weight=weight)
        super().__init__(defaults, end=end)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], end: list[torch.Tensor]):
        torch._foreach_lerp_(update, end, weight=self.get_settings(objective.params, 'weight'))
        return update

class CopySign(BinaryOperationBase):
    """Returns tensors with sign copied from ``other(tensors)``."""
    def __init__(self, other: Chainable):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], other: list[torch.Tensor]):
        return [u.copysign_(o) for u, o in zip(update, other)]

class RCopySign(BinaryOperationBase):
    """Returns ``other(tensors)`` with sign copied from tensors."""
    def __init__(self, other: Chainable):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], other: list[torch.Tensor]):
        return [o.copysign_(u) for u, o in zip(update, other)]
CopyMagnitude = RCopySign

class Clip(BinaryOperationBase):
    """clip tensors to be in  ``(min, max)`` range. ``min`` and ``max`: can be None, numbers or modules.

    If ``min`` and ``max``  are modules, this calculates ``tensors.clip(min(tensors), max(tensors))``.
    """
    def __init__(self, min: float | Chainable | None = None, max: float | Chainable | None = None):
        super().__init__({}, min=min, max=max)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], min: float | list[torch.Tensor] | None, max: float | list[torch.Tensor] | None):
        return TensorList(update).clamp_(min=min,  max=max)

class MirroredClip(BinaryOperationBase):
    """clip tensors to be in  ``(-value, value)`` range. ``value`` can be a number or a module.

    If ``value`` is a module, this calculates ``tensors.clip(-value(tensors), value(tensors))``
    """
    def __init__(self, value: float | Chainable):
        super().__init__({}, value=value)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], value: float | list[torch.Tensor]):
        min = -value if isinstance(value, (int,float)) else [-v for v in value]
        return TensorList(update).clamp_(min=min,  max=value)

class GraftInputToOutput(BinaryOperationBase):
    """Outputs ``tensors`` rescaled to have the same norm as ``magnitude(tensors)``."""
    def __init__(self, magnitude: Chainable, tensorwise:bool=True, ord:float=2, eps:float = 1e-6):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, magnitude=magnitude)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], magnitude: list[torch.Tensor]):
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(self.defaults)
        return TensorList(update).graft_(magnitude, tensorwise=tensorwise, ord=ord, eps=eps)

class GraftOutputToInput(BinaryOperationBase):
    """Outputs ``magnitude(tensors)`` rescaled to have the same norm as ``tensors``"""

    def __init__(self, direction: Chainable, tensorwise:bool=True, ord:float=2, eps:float = 1e-6):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, direction=direction)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], direction: list[torch.Tensor]):
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(self.defaults)
        return TensorList(direction).graft_(update, tensorwise=tensorwise, ord=ord, eps=eps)

class Maximum(BinaryOperationBase):
    """Outputs ``maximum(tensors, other(tensors))``"""
    def __init__(self, other: Chainable):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], other: list[torch.Tensor]):
        torch._foreach_maximum_(update, other)
        return update

class Minimum(BinaryOperationBase):
    """Outputs ``minimum(tensors, other(tensors))``"""
    def __init__(self, other: Chainable):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], other: list[torch.Tensor]):
        torch._foreach_minimum_(update, other)
        return update


class GramSchimdt(BinaryOperationBase):
    """outputs tensors made orthogonal to ``other(tensors)`` via Gram-Schmidt."""
    def __init__(self, other: Chainable):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], other: list[torch.Tensor]):
        update = TensorList(update); other = TensorList(other)
        min = torch.finfo(update[0].dtype).tiny * 2
        return update - (other*update) / (other*other).clip(min=min)


class Threshold(BinaryOperationBase):
    """Outputs tensors thresholded such that values above ``threshold`` are set to ``value``."""
    def __init__(self, threshold: Chainable | float, value: Chainable | float, update_above: bool):
        defaults = dict(update_above=update_above)
        super().__init__(defaults, threshold=threshold, value=value)

    @torch.no_grad
    def transform(self, objective, update: list[torch.Tensor], threshold: list[torch.Tensor] | float, value: list[torch.Tensor] | float):
        update_above = self.defaults['update_above']
        update = TensorList(update)
        if update_above:
            if isinstance(value, list): return update.where(update>threshold, value)
            return update.masked_fill_(update<=threshold, value)

        if isinstance(value, list): return update.where(update<threshold, value)
        return update.masked_fill_(update>=threshold, value)