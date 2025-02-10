from collections.abc import Iterable
from operator import methodcaller

import torch

from ...core import OptimizerModule
from ...tensorlist import TensorList


class Operation(OptimizerModule):
    """Applies an operation to the ascent, supported operations:

    `abs`, `sign`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`,
    `tanh`, `log`, `log1p`, `log2`, `log10`, `erf`, `erfc`, `exp`, `neg`, `reciprocal`,
    `copy`, `zero`, `sqrt`, `floor`, `ceil`, `round`."""
    def __init__(self, operation: str):
        super().__init__({})
        self.operation = methodcaller(f'{operation}_')

    @torch.no_grad
    def _update(self, vars, ascent): return self.operation(ascent)

class Reciprocal(OptimizerModule):
    """*1 / update*"""
    def __init__(self,):
        super().__init__({})

    @torch.no_grad()
    def _update(self, vars, ascent): return ascent.reciprocal_()

class Negate(OptimizerModule):
    """minus update"""
    def __init__(self,):
        super().__init__({})

    @torch.no_grad()
    def _update(self, vars, ascent): return ascent.neg_()


def sign_grad_(params: Iterable[torch.Tensor]):
    """Apply sign function to gradients of an iterable of parameters.

    Args:
        params (abc.Iterable[torch.Tensor]): an iterable of Tensors or a single Tensor.
    """
    TensorList(params).get_existing_grads().sign_()

class Sign(OptimizerModule):
    """applies sign function to the update"""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def _update(self, vars, ascent): return ascent.sign_()

class Abs(OptimizerModule):
    """takes absolute values of the update."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def _update(self, vars, ascent): return ascent.abs_()

class Sin(OptimizerModule):
    """applies sin function to the ascent"""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def _update(self, vars, ascent): return ascent.sin_()

class Cos(OptimizerModule):
    """applies cos function to the ascent"""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def _update(self, vars, ascent): return ascent.cos_()


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
    def _update(self, vars, ascent): return ascent.nan_to_num_(self.nan, self.posinf, self.neginf)


class MagnitudePower(OptimizerModule):
    """Raises update to the `value` power, but preserves the sign when the power is odd."""
    def __init__(self, value: int | float):
        super().__init__({})
        self.value = value

    @torch.no_grad()
    def _update(self, vars, ascent):
        if self.value % 2 == 1: return ascent.pow_(self.value)
        return ascent.abs().pow_(self.value) * ascent.sign()

