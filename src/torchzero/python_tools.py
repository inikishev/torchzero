import functools
import operator
import typing as T
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import torch

def _flatten_no_check(iterable: Iterable) -> list[T.Any]:
    """Flatten an iterable of iterables, returns a flattened list. Note that if `iterable` is not Iterable, this will return `[iterable]`."""
    if isinstance(iterable, Iterable):
        return [a for i in iterable for a in _flatten_no_check(i)]
    else:
        return [iterable]

def flatten(iterable: Iterable) -> list[T.Any]:
    """Flatten an iterable of iterables, returns a flattened list. If `iterable` is not iterable, raises a TypeError."""
    if isinstance(iterable, Iterable): return [a for i in iterable for a in _flatten_no_check(i)]
    else: raise TypeError(f'passed object is not an iterable, {type(iterable) = }')

def reduce_dim[X](x:Iterable[Iterable[X]]) -> list[X]: # pylint:disable=E0602
    """Reduces one level of nesting. Takes an iterable of iterables of X, and returns an iterable of X."""
    return functools.reduce(operator.iconcat, x, [])

ScalarType = float | torch.Tensor