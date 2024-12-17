import torch
from torchzero.tensorlist import TensorList
from collections.abc import Callable
from ...core import OptimizerModule, _get_loss


class Clone(OptimizerModule):
    def __init__(self):
        """Clones the update. Some modules update ascent in-place, so this may be
        useful if you need to preserve it."""
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent): return ascent.clone()
class Noop(OptimizerModule):
    def __init__(self, *args, **kwargs):
        """does nothing"""
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent): return ascent
class Lambda(OptimizerModule):
    def __init__(self, f: Callable[[TensorList], TensorList]):
        """Applies a function to the ascent direction. The function must take a TensorList as the argument.

        Args:
            f (_type_): function
        """
        super().__init__({})
        self.f = f

    @torch.no_grad()
    def _update(self, state, ascent): return self.f(ascent)

class Reciprocal(OptimizerModule):
    def __init__(self,):
        """Calculates reciprocal of the update (1 / update)."""
        super().__init__({})

    @torch.no_grad()
    def _update(self, state, ascent): return ascent.reciprocal_()

class Add(OptimizerModule):
    def __init__(self, value):
        """Adds `value` to the update."""
        super().__init__({})
        self.value = value
    @torch.no_grad()
    def _update(self, state, ascent): return ascent.add_(self.value)

class AddSign(OptimizerModule):
    def __init__(self, value):
        """Add `value` multiplied by sign of the ascent, i.e. this adds `value` to the magnitude of the update."""
        super().__init__({})
        self.value = value
    @torch.no_grad()
    def _update(self, state, ascent):
        return ascent.add_(ascent.clamp_magnitude(min=1).sign_().mul_(self.value))

class Mul(OptimizerModule):
    def __init__(self, value):
        super().__init__({})
        self.value = value
    @torch.no_grad()
    def _update(self, state, ascent) -> TensorList:return ascent.mul_(self.value)

class Div(OptimizerModule):
    def __init__(self, value):
        """divides update by `value`."""
        super().__init__({})
        self.value = value
    @torch.no_grad()
    def _update(self, state, ascent) -> TensorList: return ascent.div_(self.value)


class Pow(OptimizerModule):
    def __init__(self, value):
        """applies power to the update."""
        super().__init__({})
        self.value = value
    @torch.no_grad()
    def _update(self, state, ascent): return ascent.pow_(self.value)

class PowSign(OptimizerModule):
    def __init__(self, value):
        """Applies power to the update, but preserves the sign."""
        super().__init__({})
        self.value = value
    @torch.no_grad()
    def _update(self, state, ascent):
        if self.value % 2 == 1: return ascent.pow_(self.value)
        return ascent.abs().pow_(self.value) * ascent.sign()

class NanToNum(OptimizerModule):
    def __init__(self, nan=None, posinf=None, neginf=None):
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
        super().__init__({})
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf

    @torch.no_grad()
    def _update(self, state, ascent): return ascent.nan_to_num_(self.nan, self.posinf, self.neginf)