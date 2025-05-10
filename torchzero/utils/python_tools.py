import functools
import operator
from typing import Any, TypeVar
from collections.abc import Iterable, Callable
from collections import UserDict


def _flatten_no_check(iterable: Iterable) -> list[Any]:
    """Flatten an iterable of iterables, returns a flattened list. Note that if `iterable` is not Iterable, this will return `[iterable]`."""
    if isinstance(iterable, Iterable) and not isinstance(iterable, str):
        return [a for i in iterable for a in _flatten_no_check(i)]
    return [iterable]

def flatten(iterable: Iterable) -> list[Any]:
    """Flatten an iterable of iterables, returns a flattened list. If `iterable` is not iterable, raises a TypeError."""
    if isinstance(iterable, Iterable): return [a for i in iterable for a in _flatten_no_check(i)]
    raise TypeError(f'passed object is not an iterable, {type(iterable) = }')

X = TypeVar("X")
# def reduce_dim[X](x:Iterable[Iterable[X]]) -> list[X]: # pylint:disable=E0602
def reduce_dim(x:Iterable[Iterable[X]]) -> list[X]: # pylint:disable=E0602
    """Reduces one level of nesting. Takes an iterable of iterables of X, and returns an iterable of X."""
    return functools.reduce(operator.iconcat, x, [])

def generic_eq(x: int | float | Iterable[int | float], y: int | float | Iterable[int | float]) -> bool:
    """generic equals function that supports scalars and lists of numbers"""
    if isinstance(x, (int,float)):
        if isinstance(y, (int,float)): return x==y
        return all(i==x for i in y)
    if isinstance(y, (int,float)):
        return all(i==y for i in x)
    return all(i==j for i,j in zip(x,y))

def zipmap(self, fn: Callable, other: Any | list | tuple, *args, **kwargs):
    """If `other` is list/tuple, applies `fn` to self zipped with `other`.
    Otherwise applies `fn` to this sequence and `other`.
    Returns a new sequence with return values of the callable."""
    if isinstance(other, (list, tuple)): return self.__class__(fn(i, j, *args, **kwargs) for i, j in zip(self, other))
    return self.__class__(fn(i, other, *args, **kwargs) for i in self)


K = TypeVar("K")
V = TypeVar("V")
class FallbackDict(UserDict[K, V]):
    """Dictionary with a fallback dictionary.

    :code:`fd = d[key]` - returns data[key] if key is in data else fallback[key]

    :code:`fd[key] = value` - always sets data[key] = value

    :code:`fd.keys(); fd.values(); fd(items); dict(fd); str(fd)` - for all purposes this behaves like a union of fallback and data, where data has a higher priority.

    Args:
        data (dict[str, Any]): main dict.
        fallback (dict[str, Any]): fallback dict.
    """
    __slots__ = ('fallback', )
    def __init__(self, data: dict[K, V], fallback: dict[K, V]):
        super().__init__(data)
        self.fallback = fallback

    def __getitem__(self, k):
        if k in self.data: return self.data[k]
        return self.fallback[k]

    def keys(self): return (self.fallback | self.data).keys()
    def values(self): return (self.fallback | self.data).values()
    def items(self): return (self.fallback | self.data).items()

    def __repr__(self):
        return dict.__repr__(self.fallback | self.data)


class StepCounter:
    def __init__(self): self.step = 0
    def increment(self): self.step += 1
    def __call__(self): return self.step
    def reset(self): self.step = 0

