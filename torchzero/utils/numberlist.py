"""A lightweight data type for a list of numbers (or anything else) with arithmetic overloads (using basic for-loops).
Subclasses list so works with torch._foreach_xxx operations."""
import builtins
from collections.abc import Callable, Sequence, Iterable, Generator, Iterator
import math
import operator
from typing import Any, Literal, TypedDict
from typing_extensions import Self, TypeAlias, Unpack

import torch
from .python_tools import zipmap

def _alpha_add(x, other, alpha):
    return x + other * alpha

def as_numberlist(x):
    if isinstance(x, NumberList): return x
    return NumberList(x)


def maybe_numberlist(x):
    if isinstance(x, (list,tuple)): return as_numberlist(x)
    return x

def _clamp(x,min,max):
    if min is not None and x < min: return min
    if max is not None and x > max: return max
    return x

class NumberList(list[int | float | Any]):
    """List of python numbers.
    Note that this only supports basic arithmetic operations that are overloaded.

    Can't use a numpy array because _foreach methods do not work with it."""
    # remove torch.Tensor from return values
    # this is no longer necessary
    # def __getitem__(self, i) -> Any:
    #     return super().__getitem__(i)

    # def __iter__(self) -> Iterator[Any]:
    #     return super().__iter__()

    def __add__(self, other: Any) -> Self: return self.add(other) # type:ignore
    def __radd__(self, other: Any) -> Self: return self.add(other)

    def __sub__(self, other: Any) -> Self: return self.sub(other)
    def __rsub__(self, other: Any) -> Self: return self.sub(other).neg()

    def __mul__(self, other: Any) -> Self: return self.mul(other) # type:ignore
    def __rmul__(self, other: Any) -> Self: return self.mul(other) # type:ignore

    def __truediv__(self, other: Any) -> Self: return self.div(other)
    def __rtruediv__(self, other: Any):
        if isinstance(other, (tuple,list)): return self.__class__(o / i for o, i in zip(self, other))
        return self.__class__(other / i for i in self)

    def __floordiv__(self, other: Any): return self.floor_divide(other)
    def __mod__(self, other: Any): return self.remainder(other)


    def __pow__(self, other: Any): return self.pow(other)
    def __rpow__(self, other: Any): return self.rpow(other)

    def __neg__(self): return self.neg()

    def __eq__(self, other: Any): return self.eq(other) # type:ignore
    def __ne__(self, other: Any): return self.ne(other) # type:ignore
    def __lt__(self, other: Any): return self.lt(other) # type:ignore
    def __le__(self, other: Any): return self.le(other) # type:ignore
    def __gt__(self, other: Any): return self.gt(other) # type:ignore
    def __ge__(self, other: Any): return self.ge(other) # type:ignore

    def __invert__(self): return self.logical_not()

    def __and__(self, other: Any): return self.logical_and(other)
    def __or__(self, other: Any): return self.logical_or(other)
    def __xor__(self, other: Any): return self.logical_xor(other)

    def __bool__(self):
        raise RuntimeError(f'Boolean value of {self.__class__.__name__} is ambiguous')

    def zipmap(self, fn: Callable, other: Any | list | tuple, *args, **kwargs):
        """If `other` is list/tuple, applies `fn` to this TensorList zipped with `other`.
        Otherwise applies `fn` to this TensorList and `other`.
        Returns a new TensorList with return values of the callable."""
        return zipmap(self, fn, other, *args, **kwargs)

    def zipmap_args(self, fn: Callable[..., Any], *others, **kwargs):
        """If `args` is list/tuple, applies `fn` to this TensorList zipped with `others`.
        Otherwise applies `fn` to this TensorList and `other`."""
        others = [i if isinstance(i, (list, tuple)) else [i]*len(self) for i in others]
        return self.__class__(fn(*z, **kwargs) for z in zip(self, *others))

    # def _set_to_method_result_(self, method: str, *args, **kwargs):
    #     """Sets each element of the tensorlist to the result of calling the specified method on the corresponding element.
    #     This is used to support/mimic in-place operations, although I decided to remove them."""
    #     res = getattr(self, method)(*args, **kwargs)
    #     for i,v in enumerate(res): self[i] = v
    #     return self

    def add(self, other: Any, alpha: int | float = 1):
        if alpha == 1: return self.zipmap(operator.add, other=other)
        return self.zipmap(_alpha_add, other=other, alpha = alpha)

    def sub(self, other: Any, alpha: int | float = 1):
        if alpha == 1: return self.zipmap(operator.sub, other=other)
        return self.zipmap(_alpha_add, other=other, alpha = -alpha)

    def neg(self): return self.__class__(-i for i in self)
    def mul(self, other: Any): return self.zipmap(operator.mul, other=other)
    def div(self, other: Any) -> Self: return self.zipmap(operator.truediv, other=other)
    def pow(self, exponent: Any): return self.zipmap(math.pow, other=exponent)
    def floor_divide(self, other: Any): return self.zipmap(operator.floordiv, other=other)
    def remainder(self, other: Any): return self.zipmap(operator.mod, other=other)
    def rpow(self, other: Any): return self.zipmap(lambda x,y: y**x, other=other)

    def fill_none(self, value):
        if isinstance(value, (list,tuple)): return self.__class__(v if s is None else s for s, v in zip(self, value))
        return self.__class__(value if s is None else s for s in self)

    def logical_not(self): return self.__class__(not i for i in self)
    def logical_and(self, other: Any): return self.zipmap(operator.and_, other=other)
    def logical_or(self, other: Any): return self.zipmap(operator.or_, other=other)
    def logical_xor(self, other: Any): return self.zipmap(operator.xor, other=other)

    def map(self, fn: Callable[..., torch.Tensor], *args, **kwargs):
        """Applies `fn` to all elements of this TensorList
        and returns a new TensorList with return values of the callable."""
        return self.__class__(fn(i, *args, **kwargs) for i in self)

    def clamp(self, min=None, max=None):
        return self.zipmap_args(_clamp, min, max)
    def clip(self, min=None, max=None):
        return self.zipmap_args(_clamp, min, max)