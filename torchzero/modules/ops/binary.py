#pyright: reportIncompatibleMethodOverride=false
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from operator import itemgetter
from typing import Any

import torch

from ...core import Chainable, Module, Target, Var, maybe_chain
from ...utils import TensorList, tensorlist


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
    def transform(self, var: Var, update: list[torch.Tensor], **operands: Any | list[torch.Tensor]) -> Iterable[torch.Tensor]:
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

        transformed = self.transform(var, update=var.get_update(), **processed_operands)
        var.update = list(transformed)
        return var


class Add(BinaryOperationBase):
    """Add :code:`other` to tensors. :code:`other` can be a number or a module.

    If :code:`other` is a module, this calculates :code:`tensors + other(tensors)`
    """
    def __init__(self, other: Chainable | float, alpha: float = 1):
        defaults = dict(alpha=alpha)
        super().__init__(defaults, other=other)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], other: float | list[torch.Tensor]):
        if isinstance(other, (int,float)): torch._foreach_add_(update, other * self.settings[var.params[0]]['alpha'])
        else: torch._foreach_add_(update, other, alpha=self.settings[var.params[0]]['alpha'])
        return update

class Sub(BinaryOperationBase):
    """Subtract :code:`other` from tensors. :code:`other` can be a number or a module.

    If :code:`other` is a module, this calculates :code:`tensors - other(tensors)`
    """
    def __init__(self, other: Chainable | float, alpha: float = 1):
        defaults = dict(alpha=alpha)
        super().__init__(defaults, other=other)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], other: float | list[torch.Tensor]):
        if isinstance(other, (int,float)): torch._foreach_sub_(update, other * self.settings[var.params[0]]['alpha'])
        else: torch._foreach_sub_(update, other, alpha=self.settings[var.params[0]]['alpha'])
        return update

class RSub(BinaryOperationBase):
    """Subtract tensors from :code:`other`. :code:`other` can be a number or a module.

    If :code:`other` is a module, this calculates :code:`other(tensors) - tensors`
    """
    def __init__(self, other: Chainable | float):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], other: float | list[torch.Tensor]):
        return other - TensorList(update)

class Mul(BinaryOperationBase):
    """Multiply tensors by :code:`other`. :code:`other` can be a number or a module.

    If :code:`other` is a module, this calculates :code:`tensors * other(tensors)`
    """
    def __init__(self, other: Chainable | float):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], other: float | list[torch.Tensor]):
        torch._foreach_mul_(update, other)
        return update

class Div(BinaryOperationBase):
    """Divide tensors by :code:`other`. :code:`other` can be a number or a module.

    If :code:`other` is a module, this calculates :code:`tensors / other(tensors)`
    """
    def __init__(self, other: Chainable | float):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], other: float | list[torch.Tensor]):
        torch._foreach_div_(update, other)
        return update

class RDiv(BinaryOperationBase):
    """Divide :code:`other` by tensors. :code:`other` can be a number or a module.

    If :code:`other` is a module, this calculates :code:`other(tensors) / tensors`
    """
    def __init__(self, other: Chainable | float):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], other: float | list[torch.Tensor]):
        return other / TensorList(update)

class Pow(BinaryOperationBase):
    """Take tensors to the power of :code:`exponent`. :code:`exponent` can be a number or a module.

    If :code:`exponent` is a module, this calculates :code:`tensors ^ exponent(tensors)`
    """
    def __init__(self, exponent: Chainable | float):
        super().__init__({}, exponent=exponent)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], exponent: float | list[torch.Tensor]):
        torch._foreach_pow_(update, exponent)
        return update

class RPow(BinaryOperationBase):
    """Take :code:`other` to the power of tensors. :code:`other` can be a number or a module.

    If :code:`other` is a module, this calculates :code:`other(tensors) ^ tensors`
    """
    def __init__(self, other: Chainable | float):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], other: float | list[torch.Tensor]):
        if isinstance(other, (int, float)): return torch._foreach_pow(other, update) # no in-place
        torch._foreach_pow_(other, update)
        return other

class Lerp(BinaryOperationBase):
    """Does a linear interpolation of tensors and :code:`end` module based on a scalar :code:`weight`.

    The output is given by :code:`output = tensors + weight * (end(tensors) - tensors)`
    """
    def __init__(self, end: Chainable, weight: float):
        defaults = dict(weight=weight)
        super().__init__(defaults, end=end)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], end: list[torch.Tensor]):
        torch._foreach_lerp_(update, end, weight=self.get_settings(var.params, 'weight'))
        return update

class CopySign(BinaryOperationBase):
    """Returns tensors with sign copied from :code:`other(tensors)`."""
    def __init__(self, other: Chainable):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], other: list[torch.Tensor]):
        return [u.copysign_(o) for u, o in zip(update, other)]

class RCopySign(BinaryOperationBase):
    """Returns :code:`other(tensors)` with sign copied from tensors."""
    def __init__(self, other: Chainable):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], other: list[torch.Tensor]):
        return [o.copysign_(u) for u, o in zip(update, other)]
CopyMagnitude = RCopySign

class Clip(BinaryOperationBase):
    """clip tensors to be in  :code:`(min, max)` range. :code:`min` and :code:`max: can be None, numbers or modules.

    If code:`min` and :code:`max`:  are modules, this calculates :code:`tensors.clip(min(tensors), max(tensors))`.
    """
    def __init__(self, min: float | Chainable | None = None, max: float | Chainable | None = None):
        super().__init__({}, min=min, max=max)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], min: float | list[torch.Tensor] | None, max: float | list[torch.Tensor] | None):
        return TensorList(update).clamp_(min=min,  max=max)

class MirroredClip(BinaryOperationBase):
    """clip tensors to be in  :code:`(-value, value)` range. :code:`value` can be a number or a module.

    If :code:`value` is a module, this calculates :code:`tensors.clip(-value(tensors), value(tensors))`
    """
    def __init__(self, value: float | Chainable):
        super().__init__({}, value=value)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], value: float | list[torch.Tensor]):
        min = -value if isinstance(value, (int,float)) else [-v for v in value]
        return TensorList(update).clamp_(min=min,  max=value)

class Graft(BinaryOperationBase):
    """Outputs tensors rescaled to have the same norm as :code:`magnitude(tensors)`."""
    def __init__(self, magnitude: Chainable, tensorwise:bool=True, ord:float=2, eps:float = 1e-6):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, magnitude=magnitude)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], magnitude: list[torch.Tensor]):
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(self.settings[var.params[0]])
        return TensorList(update).graft_(magnitude, tensorwise=tensorwise, ord=ord, eps=eps)

class RGraft(BinaryOperationBase):
    """Outputs :code:`magnitude(tensors)` rescaled to have the same norm as tensors"""

    def __init__(self, direction: Chainable, tensorwise:bool=True, ord:float=2, eps:float = 1e-6):
        defaults = dict(tensorwise=tensorwise, ord=ord, eps=eps)
        super().__init__(defaults, direction=direction)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], direction: list[torch.Tensor]):
        tensorwise, ord, eps = itemgetter('tensorwise','ord','eps')(self.settings[var.params[0]])
        return TensorList(direction).graft_(update, tensorwise=tensorwise, ord=ord, eps=eps)

GraftToUpdate = RGraft

class Maximum(BinaryOperationBase):
    """Outputs :code:`maximum(tensors, other(tensors))`"""
    def __init__(self, other: Chainable):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], other: list[torch.Tensor]):
        torch._foreach_maximum_(update, other)
        return update

class Minimum(BinaryOperationBase):
    """Outputs :code:`minimum(tensors, other(tensors))`"""
    def __init__(self, other: Chainable):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], other: list[torch.Tensor]):
        torch._foreach_minimum_(update, other)
        return update


class GramSchimdt(BinaryOperationBase):
    """outputs tensors made orthogonal to `other(tensors)` via Gram-Schmidt."""
    def __init__(self, other: Chainable):
        super().__init__({}, other=other)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], other: list[torch.Tensor]):
        update = TensorList(update); other = TensorList(other)
        return update - (other*update) / ((other*other) + 1e-8)


class Threshold(BinaryOperationBase):
    """Outputs tensors thresholded such that values above :code:`threshold` are set to :code:`value`."""
    def __init__(self, threshold: Chainable | float, value: Chainable | float, update_above: bool):
        defaults = dict(update_above=update_above)
        super().__init__(defaults, threshold=threshold, value=value)

    @torch.no_grad
    def transform(self, var, update: list[torch.Tensor], threshold: list[torch.Tensor] | float, value: list[torch.Tensor] | float):
        update_above = self.settings[var.params[0]]['update_above']
        update = TensorList(update)
        if update_above:
            if isinstance(value, list): return update.where_(update>threshold, value)
            return update.masked_fill_(update<=threshold, value)

        if isinstance(value, list): return update.where_(update<threshold, value)
        return update.masked_fill_(update>=threshold, value)