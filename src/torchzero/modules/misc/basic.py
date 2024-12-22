from collections.abc import Callable, Iterable

import torch

from torchzero.tensorlist import TensorList

from ...core import OptimizerModule

class LR(OptimizerModule):
    """Multiplies update by the learning rate."""
    def __init__(self, lr = 1e-3):
        defaults = dict(lr = lr)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent):
        # multiply ascent direction by lr in-place
        lr = self.get_group_key('lr')
        ascent *= lr
        return ascent


class Clone(OptimizerModule):
    """Clones the update. Some modules update ascent in-place, so this may be
    useful if you need to preserve it."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent): return ascent.clone()

class Identity(OptimizerModule):
    """Does nothing."""
    def __init__(self, *args, **kwargs):
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent): return ascent

class Lambda(OptimizerModule):
    """Applies a function to the ascent direction.
    The function must take a TensorList as the argument, and return the modified tensorlist.

    Args:
        f (Callable): function
    """
    def __init__(self, f: Callable[[TensorList], TensorList]):
        super().__init__({})
        self.f = f

    @torch.no_grad()
    def _update(self, state, ascent): return self.f(ascent)

class Reciprocal(OptimizerModule):
    """Calculates reciprocal of the update (1 / update)."""
    def __init__(self,):
        super().__init__({})

    @torch.no_grad()
    def _update(self, state, ascent): return ascent.reciprocal_()

class Negate(OptimizerModule):
    """Negates the update (-update)"""
    def __init__(self,):
        super().__init__({})

    @torch.no_grad()
    def _update(self, state, ascent): return ascent.neg_()

class Add(OptimizerModule):
    """Adds `value` to the update."""
    def __init__(self, value):
        super().__init__({})
        self.value = value
    @torch.no_grad()
    def _update(self, state, ascent): return ascent.add_(self.value)

class AddMagnitude(OptimizerModule):
    """Add `value` multiplied by sign of the ascent, i.e. this adds `value` to the magnitude of the update.

    Args:
        value (_type_): value to add to magnitude.
        add_to_zero (bool, optional): if True, adds `value` to 0s. Otherwise, zeros remain zero. Defaults to True.
    """
    def __init__(self, value, add_to_zero=True):
        super().__init__({})
        self.value = value
        self.add_to_zero = add_to_zero
    @torch.no_grad()
    def _update(self, state, ascent):
        if self.add_to_zero: return ascent.add_(ascent.clamp_magnitude(min=1).sign_().mul_(self.value))
        return ascent.add_(ascent.sign_().mul_(self.value))

class Mul(OptimizerModule):
    """Multiplies the update by `value`."""
    def __init__(self, value):
        super().__init__({})
        self.value = value
    @torch.no_grad()
    def _update(self, state, ascent) -> TensorList:return ascent.mul_(self.value)

class Div(OptimizerModule):
    """Divides update by `value`."""
    def __init__(self, value):
        super().__init__({})
        self.value = value
    @torch.no_grad()
    def _update(self, state, ascent) -> TensorList: return ascent.div_(self.value)


class Pow(OptimizerModule):
    """Raises update to the `value` power."""
    def __init__(self, value):
        super().__init__({})
        self.value = value
    @torch.no_grad()
    def _update(self, state, ascent): return ascent.pow_(self.value)

class PowMagnitude(OptimizerModule):
    """Raises update to the `value` power, but preserves the sign."""
    def __init__(self, value):
        super().__init__({})
        self.value = value
    @torch.no_grad()
    def _update(self, state, ascent):
        if self.value % 2 == 1: return ascent.pow_(self.value)
        return ascent.abs().pow_(self.value) * ascent.sign()

class NanToNum(OptimizerModule):
    """Convert `nan`, `inf` and `-inf` to numbers.

    Args:
        nan (optional): the value to replace NaNs with. Default is zero.
        posinf (optional): if a Number, the value to replace positive infinity values with.
            If None, positive infinity values are replaced with the greatest finite value
            representable by input's dtype. Default is None.
        neginf (optional): if a Number, the value to replace negative infinity values with.
            If None, negative infinity values are replaced with the lowest finite value
            representable by input's dtype. Default is None.
    """
    def __init__(self, nan=None, posinf=None, neginf=None):
        super().__init__({})
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf

    @torch.no_grad()
    def _update(self, state, ascent): return ascent.nan_to_num_(self.nan, self.posinf, self.neginf)



def sign_grad_(params: Iterable[torch.Tensor]):
    """Apply sign function to gradients of an iterable of parameters.

    Args:
        params (abc.Iterable[torch.Tensor]): an iterable of Tensors or a single Tensor.
    """
    TensorList(params).get_existing_grads().sign_()

class Sign(OptimizerModule):
    """Applies sign function to the update"""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent):
        ascent.sign_()
        return ascent
