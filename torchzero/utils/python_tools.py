import importlib
import functools
import operator
from typing import Any, TypeVar, overload
from collections.abc import Iterable, Callable, Mapping, MutableSequence
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
# def reduce_dim[X](x:Iterable[Iterable[X]]) -> list[X]:
def reduce_dim(x:Iterable[Iterable[X]]) -> list[X]:
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

def generic_ne(x: int | float | Iterable[int | float], y: int | float | Iterable[int | float]) -> bool:
    """generic not equals function that supports scalars and lists of numbers. Faster than not generic_eq"""
    if isinstance(x, (int,float)):
        if isinstance(y, (int,float)): return x!=y
        return any(i!=x for i in y)
    if isinstance(y, (int,float)):
        return any(i!=y for i in x)
    return any(i!=j for i,j in zip(x,y))

def generic_is_none(x: Any | Iterable[Any]):
    """returns True if x is None or iterable with all elements set to None"""
    if x is None: return True
    if isinstance(x, Iterable): return all(i is None for i in x)
    return False

def zipmap(self, fn: Callable, other: Any | list | tuple, *args, **kwargs):
    """If `other` is list/tuple, applies `fn` to self zipped with `other`.
    Otherwise applies `fn` to this sequence and `other`.
    Returns a new sequence with return values of the callable."""
    if isinstance(other, (list, tuple)): return self.__class__(fn(i, j, *args, **kwargs) for i, j in zip(self, other))
    return self.__class__(fn(i, other, *args, **kwargs) for i in self)

ListLike = TypeVar('ListLike', bound=MutableSequence)
@overload
def unpack_dicts(dicts: Iterable[Mapping[str, Any]], key:str, *, cls:type[ListLike]=list) -> ListLike: ...
@overload
def unpack_dicts(dicts: Iterable[Mapping[str, Any]], key:str, key2: str, *keys:str, cls:type[ListLike]=list) -> list[ListLike]: ...
def unpack_dicts(dicts: Iterable[Mapping[str, Any]], key:str, key2: str | None = None, *keys:str, cls:type[ListLike]=list) -> ListLike | list[ListLike]:
    k1 = (key,) if isinstance(key, str) else tuple(key)
    k2 = () if key2 is None else (key2,)
    keys = k1 + k2 + keys

    values = [cls(s[k] for s in dicts) for k in keys] # pyright:ignore[reportCallIssue]
    if len(values) == 1: return values[0]
    return values


def safe_dict_update_(d1_:dict, d2:dict):
    inter = set(d1_.keys()).intersection(d2.keys())
    if len(inter) > 0: raise RuntimeError(f"Duplicate keys {inter}")
    d1_.update(d2)

# lazy loader from https://stackoverflow.com/a/78312674/15673832
class LazyLoader:
    'thin shell class to wrap modules.  load real module on first access and pass thru'

    def __init__(self, modname):
        self._modname = modname
        self._mod = None

    def __getattr__(self, attr):
        'import module on first attribute access'

        try:
            return getattr(self._mod, attr)

        except Exception as e :
            if self._mod is None :
                # module is unset, load it
                self._mod = importlib.import_module (self._modname)
            else :
                # module is set, got different exception from getattr ().  reraise it
                raise e

        # retry getattr if module was just loaded for first time
        # call this outside exception handler in case it raises new exception
        return getattr (self._mod, attr)
