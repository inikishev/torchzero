# pyright: reportIncompatibleMethodOverride=false
r"""
TensorList is a data type that can be used to manipulate a sequence of tensors such as model parameters,
with the same methods that normal tensors have, plus some additional convenience features.
Whenever possible, I used _foreach methods and other tricks to speed up computation.

TensorList is similar to TensorDict (https://github.com/pytorch/tensordict).
If you want to get the most performance out of a collection of tensors, use TensorDict and lock it.
However I found that *creating* a TensorDict is quite slow. In fact it negates the benefits of using it
in an optimizer when you have to create one from parameters on each step. The solution could be to create
it once beforehand, but then you won't be able to easily support parameter groups and per-parameter states.
"""
import builtins
import math
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from typing import Any, Literal, TypedDict, overload

import torch
from typing_extensions import Self, TypeAlias, Unpack

from .metrics import Metrics, evaluate_metric, calculate_metric_list
from .numberlist import NumberList, as_numberlist, maybe_numberlist
from .python_tools import generic_ne, zipmap

_Scalar = int | float | bool | complex
_TensorSeq = list[torch.Tensor] | tuple[torch.Tensor, ...]
_ScalarSeq = list[int] | list[float] | list[bool] | list[complex] | tuple[int] | tuple[float] | tuple[bool] | tuple[complex]
_ScalarSequence = Sequence[_Scalar] # i only check (list,tuple), its faster and safer
_STSeq = _TensorSeq | _ScalarSeq
_STOrSTSeq = _Scalar | torch.Tensor | _ScalarSeq | _TensorSeq

_Dim = int | list[int] | tuple[int,...] | Literal['global'] | None

Distributions = Literal['normal', 'gaussian', 'uniform', 'sphere', 'rademacher']

class _NewTensorKwargs(TypedDict, total = False):
    memory_format: Any
    dtype: Any
    layout: Any
    device: Any
    pin_memory: bool
    requires_grad: bool

# _foreach_methods = {attr.replace('_foreach_', ''):getattr(torch, attr) for attr in dir(torch) if attr.startswith('_foreach_')}
class _MethodCallerWithArgs:
    """Return a callable object that calls the given method on its operand.

    This is similar to operator.methodcaller but args and kwargs are specificed in __call__.

    Args:
        method (str): name of method to call.
    """
    __slots__ = ('_name',)
    def __init__(self, name: str):
        self._name = name

    def __call__(self, obj, *args, **kwargs):
        return getattr(obj, self._name)(*args, **kwargs)

    def __repr__(self):
        return f'{self.__class__.__module__}.{self.__class__.__name__}({repr(self._name)})'

    def __reduce__(self):
        return self.__class__, self._name

def as_tensorlist(x):
    if isinstance(x, TensorList): return x
    return TensorList(x)


# tensorlist must subclass list
# UserList doesn't work with _foreach_xxx
class TensorList(list[torch.Tensor | Any]):
    @classmethod
    def complex(cls, real: _TensorSeq, imag: _TensorSeq):
        """Create a complex TensorList from real and imaginary tensor sequences."""
        return cls(torch.complex(r, i) for r, i in zip(real, imag))

    @property
    def device(self): return [i.device for i in self]
    @property
    def dtype(self): return [i.dtype for i in self]
    @property
    def requires_grad(self): return [i.requires_grad for i in self]
    @property
    def shape(self): return [i.shape for i in self]
    def size(self, dim: int | None = None): return [i.size(dim) for i in self]
    @property
    def ndim(self): return [i.ndim for i in self]
    def ndimension(self): return [i.ndimension() for i in self]
    def numel(self): return [i.numel() for i in self]

    @property
    def grad(self): return self.__class__(i.grad for i in self)
    @property
    def real(self): return self.__class__(i.real for i in self)
    @property
    def imag(self): return self.__class__(i.imag for i in self)

    def view_as_real(self): return self.__class__(torch.view_as_real(i) for i in self)
    def view_as_complex(self): return self.__class__(torch.view_as_complex(i) for i in self)

    def type_as(self, other: torch.Tensor | _TensorSeq):
        return self.zipmap(_MethodCallerWithArgs('type_as'), other)

    def view_as(self, other: torch.Tensor | Sequence[torch.Tensor]):
        if isinstance(other, Sequence): return self.__class__(s.view_as(o) for s, o in zip(self, other))
        return self.__class__(s.view_as(other) for s in self)

    def fill_none(self, reference: Iterable[torch.Tensor]):
        """all None values are replaced with zeros of the same shape as corresponding `reference` tensor."""
        return self.__class__(t if t is not None else torch.zeros_like(r) for t,r in zip(self, reference))

    def fill_none_(self, reference: Iterable[torch.Tensor]):
        """all None values are replaced with zeros of the same shape as corresponding `reference` tensor."""
        for i, (t,r) in enumerate(zip(self, reference)):
            if t is None: self[i] = torch.zeros_like(r)
        return self

    def get_grad(self):
        """Returns all gradients that are not None."""
        return self.__class__(i.grad for i in self if i.grad is not None)

    def with_requires_grad(self, requires_grad = True):
        """Returns all tensors with requires_grad set to the given value."""
        return self.__class__(i for i in self if i.requires_grad == requires_grad)

    def with_grad(self):
        """returns all tensors whose .grad is not None"""
        return self.__class__(i for i in self if i.grad is not None)

    def ensure_grad_(self):
        """For each element, if grad is None and it requires grad, sets grad to zeroes."""
        for i in self:
            if i.requires_grad and i.grad is None: i.grad = torch.zeros_like(i)
        return self

    def accumulate_grad_(self, grads: _TensorSeq):
        """Creates grad if it is None, otherwise adds to existing grad."""
        for i, g in zip(self, grads):
            if i.grad is None: i.grad = g
            else: i.grad.add_(g)
        return self

    def set_grad_(self, grads: _TensorSeq):
        """Assings grad attributes to the given sequence, replaces grad that already exists."""
        for i, g in zip(self, grads): i.grad = g
        return self

    def zero_grad_(self, set_to_none = True):
        """Set all grads to None or zeroes."""
        if set_to_none:
            for p in self: p.grad = None
        else:
            self.get_grad().zero_()
        return self

    def __add__(self, other: _STOrSTSeq) -> Self: return self.add(other) # pyright: ignore[reportCallIssue,reportArgumentType]
    def __radd__(self, other: _STOrSTSeq) -> Self: return self.add(other) # pyright: ignore[reportCallIssue,reportArgumentType]
    def __iadd__(self, other: _STOrSTSeq) -> Self: return self.add_(other) # pyright: ignore[reportCallIssue,reportArgumentType]

    def __sub__(self, other: "_Scalar | _STSeq") -> Self: return self.sub(other) # pyright: ignore[reportCallIssue,reportArgumentType]
    def __rsub__(self, other: "_Scalar | _STSeq") -> Self: return self.sub(other).neg_() # pyright: ignore[reportCallIssue,reportArgumentType]
    def __isub__(self, other: "_Scalar | _STSeq") -> Self: return self.sub_(other) # pyright: ignore[reportCallIssue,reportArgumentType]

    def __mul__(self, other: _STOrSTSeq) -> Self: return self.mul(other)
    def __rmul__(self, other: _STOrSTSeq) -> Self: return self.mul(other)
    def __imul__(self, other: _STOrSTSeq) -> Self: return self.mul_(other)

    def __truediv__(self, other: "_STOrSTSeq") -> Self: return self.div(other)
    def __rtruediv__(self, other: "_STOrSTSeq") -> Self: return other * self.reciprocal()
    def __itruediv__(self, other: "_STOrSTSeq") -> Self: return self.div_(other)

    def __floordiv__(self, other: _STOrSTSeq): return self.floor_divide(other)
    #def __rfloordiv__(self, other: "TensorList"): return other.floor_divide(self)
    def __ifloordiv__(self, other: _STOrSTSeq): return self.floor_divide_(other)

    def __mod__(self, other: _STOrSTSeq): return self.remainder(other)
    #def __rmod__(self, other: STOrSTSequence): return self.remainder(other)
    def __imod__(self, other: _STOrSTSeq):return self.remainder_(other)

    def __pow__(self, other: "_Scalar | _STSeq"): return self.pow(other)
    def __rpow__(self, other: "_Scalar | _TensorSeq"): return self.rpow(other)
    def __ipow__(self, other: "_Scalar | _STSeq"): return self.pow_(other)

    def __neg__(self): return self.neg()

    def __eq__(self, other: _STOrSTSeq): return self.eq(other)
    def __ne__(self, other: _STOrSTSeq): return self.ne(other)
    def __lt__(self, other: _STOrSTSeq): return self.lt(other)
    def __le__(self, other: _STOrSTSeq): return self.le(other)
    def __gt__(self, other: _STOrSTSeq): return self.gt(other)
    def __ge__(self, other: _STOrSTSeq): return self.ge(other)

    def __invert__(self): return self.logical_not()

    def __and__(self, other: torch.Tensor | _TensorSeq): return self.logical_and(other)
    def __iand__(self, other: torch.Tensor | _TensorSeq): return self.logical_and_(other)
    def __or__(self, other: torch.Tensor | _TensorSeq): return self.logical_or(other)
    def __ior__(self, other: torch.Tensor | _TensorSeq): return self.logical_or_(other)
    def __xor__(self, other: torch.Tensor | _TensorSeq): return self.logical_xor(other)
    def __ixor__(self, other: torch.Tensor | _TensorSeq): return self.logical_xor_(other)

    def __bool__(self):
        raise RuntimeError(f'Boolean value of {self.__class__.__name__} is ambiguous')

    def map(self, fn: Callable[..., torch.Tensor], *args, **kwargs):
        """Applies `fn` to all elements of this TensorList
        and returns a new TensorList with return values of the callable."""
        return self.__class__(fn(i, *args, **kwargs) for i in self)
    def map_inplace_(self, fn: Callable[..., Any], *args, **kwargs):
        """Applies an in-place `fn` to all elements of this TensorList."""
        for i in self: fn(i, *args, **kwargs)
        return self

    def filter(self, fn: Callable[..., bool], *args, **kwargs):
        """Returns a TensorList with all elements for which `fn` returned True."""
        return self.__class__(i for i in self if fn(i, *args, **kwargs))

    def filter_by_list(self, s: Sequence[bool]):
        """returns a new TensorList with all elements where corresponding elements in :code:`s` are True."""
        if len(self) != len(s):
            raise ValueError(f"{len(self) = }, {len(s) = }")
        return self.__class__(i for i, boolean in zip(self, s) if boolean)

    def zipmap(self, fn: Callable, other: Any | list | tuple, *args, **kwargs):
        """If `other` is list/tuple, applies `fn` to this TensorList zipped with `other`.
        Otherwise applies `fn` to this TensorList and `other`.
        Returns a new TensorList with return values of the callable."""
        return zipmap(self, fn, other, *args, **kwargs)

    def zipmap_inplace_(self, fn: Callable[..., Any], other: Any | list | tuple, *args, **kwargs):
        """If `other` is list/tuple, applies `fn` to this TensorList zipped with `other`.
        Otherwise applies `fn` to this TensorList and `other`.
        The callable must modify elements in-place."""
        if isinstance(other, (list, tuple)):
            for i, j in zip(self, other): fn(i, j, *args, **kwargs)
        else:
            for i in self: fn(i, other, *args, **kwargs)
        return self

    def zipmap_args(self, fn: Callable[..., Any], *others, **kwargs):
        """If `args` is list/tuple, applies `fn` to this TensorList zipped with `others`.
        Otherwise applies `fn` to this TensorList and `other`."""
        others = [i if isinstance(i, (list, tuple)) else [i]*len(self) for i in others]
        return self.__class__(fn(*z, **kwargs) for z in zip(self, *others))

    def zipmap_args_inplace_(self, fn: Callable[..., Any], *others, **kwargs):
        """If `args` is list/tuple, applies `fn` to this TensorList zipped with `other`.
        Otherwise applies `fn` to this TensorList and `other`.
        The callable must modify elements in-place."""
        others = [i if isinstance(i, (list, tuple)) else [i]*len(self) for i in others]
        for z in zip(self, *others): fn(*z, **kwargs)
        return self

    def _foreach_apply(self, fn: Callable[[list[torch.Tensor]], list[torch.Tensor]], *args, **kwargs):
        """Applies a torch._foreach_xxx function to self and converts returned list back to TensorList or subclass."""
        return self.__class__(fn(self), *args, **kwargs)

    # def __getattr__(self, name: str) -> Callable:
    #     if name == '__torch_function__' or name == '_ipython_canary_method_should_not_exist_': raise AttributeError('who ？？？')
    #     if name in _foreach_methods:
    #         method = partial(self._foreach_apply, _foreach_methods[name])
    #     else:
    #         method = partial(self.map, MethodCallerWithArgs(name))
    #     setattr(self, name, method)
    #     return method

    def to(self, *args, **kwargs): return self.__class__(i.to(*args, **kwargs) for i in self)
    def cuda(self): return self.__class__(i.cuda() for i in self)
    def cpu(self): return self.__class__(i.cpu() for i in self)
    def long(self): return self.__class__(i.long() for i in self)
    def short(self): return self.__class__(i.short() for i in self)
    def clone(self): return self.__class__(i.clone() for i in self)
    def detach(self): return self.__class__(i.detach() for i in self)
    def detach_(self):
        for i in self: i.detach_()
        return self
    def contiguous(self): return self.__class__(i.contiguous() for i in self)

    # apparently I can't use float for typing if I call a method "float"
    def as_float(self): return self.__class__(i.float() for i in self)
    def as_bool(self): return self.__class__(i.bool() for i in self)
    def as_int(self): return self.__class__(i.int() for i in self)

    def copy_(self, src: _TensorSeq, non_blocking = False):
        """Copies the elements from src tensors into self tensors."""
        torch._foreach_copy_(self, src, non_blocking=non_blocking)
    def set_(self, storage: Iterable[torch.Tensor | torch.types.Storage]):
        """Sets elements of this TensorList to the values of a list of tensors."""
        for i, j in zip(self, storage): i.set_(j) # pyright:ignore[reportArgumentType]
        return self


    def requires_grad_(self, mode: bool = True):
        for e in self: e.requires_grad_(mode)
        return self

    def to_vec(self): return torch.cat(self.ravel())
    def from_vec_(self, vec:torch.Tensor):
        """Sets elements of this TensorList to the values of a 1D tensor.
        The length of the tensor must be equal to the total number of elements in this TensorList."""
        cur = 0
        for el in self:
            numel = el.numel()
            el.set_(vec[cur:cur + numel].type_as(el).view_as(el)) # pyright:ignore[reportArgumentType]
            cur += numel
        return self

    def from_vec(self, vec:torch.Tensor):
        """Creates a new TensorList from this TensorList but with values from a 1D tensor.
        The length of the tensor must be equal to the total number of elements in this TensorList."""
        res = []
        cur = 0
        for el in self:
            numel = el.numel()
            res.append(vec[cur:cur + numel].type_as(el).view_as(el))
            cur += numel
        return TensorList(res)

    # using single operation on a vec, e.g. torch.sum(self.to_vec()) can be faster but its less memory efficient
    def global_min(self) -> torch.Tensor: return builtins.min(self.min()) # pyright:ignore[reportArgumentType]
    def global_max(self) -> torch.Tensor: return builtins.max(self.max()) # pyright:ignore[reportArgumentType]
    def global_mean(self) -> torch.Tensor: return self.global_sum()/self.global_numel()
    def global_sum(self) -> torch.Tensor: return builtins.sum(self.sum()) # pyright:ignore[reportArgumentType,reportReturnType]
    def global_std(self) -> torch.Tensor: return torch.std(self.to_vec())
    def global_var(self) -> torch.Tensor: return torch.var(self.to_vec())

    def global_vector_norm(self, ord:float = 2) -> torch.Tensor:
        # return torch.linalg.vector_norm(self.to_vec(), ord = ord) # pylint:disable = not-callable
        if ord == torch.inf: return self.abs().global_max()
        if ord == -torch.inf: return self.abs().global_min()
        if ord == 1: return self.abs().global_sum()
        if ord % 2 == 0: return self.pow(ord).global_sum().pow(1/ord)
        if ord == 0: return (self != 0).global_sum().to(self[0].dtype)

        return self.abs().pow_(ord).global_sum().pow(1/ord)

    def global_metric(self, metric: Metrics) -> torch.Tensor:
        return evaluate_metric(self, metric)

    def global_any(self): return builtins.any(self.any())
    def global_all(self): return builtins.all(self.all())
    def global_numel(self) -> int: return builtins.sum(self.numel())

    def global_allclose(self, other: _TensorSeq, rtol: float = 0.00001, atol: float = 1e-8, equal_nan: bool = False) -> bool:
        bools = self.zipmap_args(torch.allclose, other, rtol, atol, equal_nan)
        return all(bools)

    def empty_like(self, **kwargs: Unpack[_NewTensorKwargs]): return self.__class__(torch.empty_like(i, **kwargs) for i in self)
    def zeros_like(self, **kwargs: Unpack[_NewTensorKwargs]): return self.__class__(torch.zeros_like(i, **kwargs) for i in self)
    def ones_like(self, **kwargs: Unpack[_NewTensorKwargs]): return self.__class__(torch.ones_like(i, **kwargs) for i in self)
    def full_like(self, fill_value: "_Scalar | _ScalarSeq", **kwargs: Unpack[_NewTensorKwargs]):
        #return self.__class__(torch.full_like(i, fill_value=fill_value, **kwargs) for i in self)
        return self.zipmap(torch.full_like, other=fill_value, **kwargs)

    def rand_like(self, generator=None, dtype: Any=None, device: Any=None, **kwargs):
        if generator is not None:
            return self.__class__(torch.rand(t.shape, generator=generator,
                                             dtype=t.dtype if dtype is None else dtype,
                                             device=t.device if device is None else device, **kwargs) for t in self)

        return self.__class__(torch.rand_like(i, dtype=dtype, device=device, **kwargs) for i in self)

    def randn_like(self, generator=None, dtype: Any=None, device: Any=None, **kwargs):

        if generator is not None:
            return self.__class__(torch.randn(t.shape, generator=generator,
                                             dtype=t.dtype if dtype is None else dtype,
                                             device=t.device if device is None else device, **kwargs) for t in self)

        return self.__class__(torch.randn_like(i, dtype=dtype, device=device, **kwargs) for i in self)

    def randint_like(self, low: "_Scalar | _ScalarSeq", high: "_Scalar | _ScalarSeq", **kwargs: Unpack[_NewTensorKwargs]):
        return self.zipmap_args(torch.randint_like, low, high, **kwargs)

    def uniform_like(self, low: "_Scalar | _ScalarSeq" = 0, high: "_Scalar | _ScalarSeq" = 1, generator=None, **kwargs: Unpack[_NewTensorKwargs]):
        res = self.empty_like(**kwargs)
        res.uniform_(low, high, generator=generator)
        return res

    def sphere_like(self, radius: "_Scalar | _ScalarSeq", generator=None, **kwargs: Unpack[_NewTensorKwargs]) -> Self:
        r = self.randn_like(generator=generator, **kwargs)
        return r.mul_(maybe_numberlist(radius) / r.global_vector_norm())

    def bernoulli(self, generator = None):
        return self.__class__(torch.bernoulli(i, generator=generator) for i in self)

    def bernoulli_like(self, p: "_Scalar | _ScalarSeq" = 0.5, generator = None, **kwargs: Unpack[_NewTensorKwargs]):
        """p is probability of a 1, other values will be 0."""
        return self.__class__(torch.bernoulli(i, generator = generator) for i in self.full_like(p, **kwargs))

    def rademacher_like(self, p: "_Scalar | _ScalarSeq" = 0.5, generator = None, **kwargs: Unpack[_NewTensorKwargs]):
        """p is probability of a 1, other values will be -1."""
        return self.bernoulli_like(p, generator=generator, **kwargs).mul_(2).sub_(1)

    def sample_like(self, distribution: Distributions = 'normal', variance: "_Scalar | _ScalarSeq | Sequence | None" = None, generator=None, **kwargs: Unpack[_NewTensorKwargs]):
        """Sample around 0."""
        if isinstance(variance, Sequence):
            if all(v is None for v in variance): variance = None
            else: variance = [v if v is not None else 1 for v in variance]

        if distribution in ('normal', 'gaussian'):
            ret = self.randn_like(generator=generator, **kwargs)
            if variance is not None: ret *= variance
            return ret

        if distribution == 'uniform':
            b = 1
            if variance is not None:
                b = ((12 * maybe_numberlist(variance)) ** 0.5) / 2
            return self.uniform_like(-b, b, generator=generator, **kwargs)

        if distribution == 'sphere':
            if variance is None: radius = 1
            else: radius = maybe_numberlist(variance) * math.sqrt(self.global_numel())
            return self.sphere_like(radius, generator=generator, **kwargs)

        if distribution == 'rademacher':
            ret = self.rademacher_like(generator=generator, **kwargs)
            if variance is not None: ret *= variance
            return ret

        raise ValueError(f'Unknow distribution {distribution}')

    def eq(self, other: _STOrSTSeq): return self.zipmap(torch.eq, other)
    def eq_(self, other: _STOrSTSeq): return self.zipmap_inplace_(_MethodCallerWithArgs('eq_'), other)
    def ne(self, other: _STOrSTSeq): return self.zipmap(torch.ne, other)
    def ne_(self, other: _STOrSTSeq): return self.zipmap_inplace_(_MethodCallerWithArgs('ne_'), other)
    def lt(self, other: _STOrSTSeq): return self.zipmap(torch.lt, other)
    def lt_(self, other: _STOrSTSeq): return self.zipmap_inplace_(_MethodCallerWithArgs('lt_'), other)
    def le(self, other: _STOrSTSeq): return self.zipmap(torch.le, other)
    def le_(self, other: _STOrSTSeq): return self.zipmap_inplace_(_MethodCallerWithArgs('le_'), other)
    def gt(self, other: _STOrSTSeq): return self.zipmap(torch.gt, other)
    def gt_(self, other: _STOrSTSeq): return self.zipmap_inplace_(_MethodCallerWithArgs('gt_'), other)
    def ge(self, other: _STOrSTSeq): return self.zipmap(torch.ge, other)
    def ge_(self, other: _STOrSTSeq): return self.zipmap_inplace_(_MethodCallerWithArgs('ge_'), other)

    def logical_and(self, other: torch.Tensor | _TensorSeq): return self.zipmap(torch.logical_and, other)
    def logical_and_(self, other: torch.Tensor | _TensorSeq): return self.zipmap_inplace_(_MethodCallerWithArgs('logical_and_'), other)
    def logical_or(self, other: torch.Tensor | _TensorSeq): return self.zipmap(torch.logical_or, other)
    def logical_or_(self, other: torch.Tensor | _TensorSeq): return self.zipmap_inplace_(_MethodCallerWithArgs('logical_or_'), other)
    def logical_xor(self, other: torch.Tensor | _TensorSeq): return self.zipmap(torch.logical_xor, other)
    def logical_xor_(self, other: torch.Tensor | _TensorSeq): return self.zipmap_inplace_(_MethodCallerWithArgs('logical_xor_'), other)

    def logical_not(self): return self.__class__(torch.logical_not(i) for i in self)
    def logical_not_(self):
        for i in self: i.logical_not_()
        return self

    def equal(self, other: torch.Tensor | _TensorSeq):
        """returns TensorList of boolean values, True if two tensors have the same size and elements, False otherwise."""
        return self.zipmap(torch.equal, other)

    @overload
    def add(self, other: torch.Tensor | _TensorSeq, alpha: _Scalar = 1): ...
    @overload
    def add(self, other: _Scalar | _ScalarSeq): ...
    def add(self, other: _STOrSTSeq, alpha: _Scalar = 1):
        if alpha == 1: return self.__class__(torch._foreach_add(self, other))
        return self.__class__(torch._foreach_add(self, other, alpha = alpha)) # pyright:ignore[reportCallIssue,reportArgumentType]

    @overload
    def add_(self, other: torch.Tensor | _TensorSeq, alpha: _Scalar = 1): ...
    @overload
    def add_(self, other: _Scalar | _ScalarSeq): ...
    def add_(self, other: _STOrSTSeq, alpha: _Scalar = 1):
        if alpha == 1: torch._foreach_add_(self, other)
        else: torch._foreach_add_(self, other, alpha = alpha) # pyright:ignore[reportCallIssue,reportArgumentType]
        return self

    def lazy_add(self, other: int | float | list[int | float] | tuple[int | float]):
        if generic_ne(other, 0): return self.add(other)
        return self
    def lazy_add_(self, other: int | float | list[int | float] | tuple[int | float]):
        if generic_ne(other, 0): return self.add_(other)
        return self

    @overload
    def sub(self, other: _TensorSeq, alpha: _Scalar = 1): ...
    @overload
    def sub(self, other: _Scalar | _ScalarSeq): ...
    def sub(self, other: "_Scalar | _STSeq", alpha: _Scalar = 1):
        if alpha == 1: return self.__class__(torch._foreach_sub(self, other))
        return self.__class__(torch._foreach_sub(self, other, alpha = alpha)) # pyright:ignore[reportArgumentType]

    @overload
    def sub_(self, other: _TensorSeq, alpha: _Scalar = 1): ...
    @overload
    def sub_(self, other: _Scalar | _ScalarSeq): ...
    def sub_(self, other: "_Scalar | _STSeq", alpha: _Scalar = 1):
        if alpha == 1: torch._foreach_sub_(self, other)
        else: torch._foreach_sub_(self, other, alpha = alpha) # pyright:ignore[reportArgumentType]
        return self

    def lazy_sub(self, other: int | float | list[int | float] | tuple[int | float]):
        if generic_ne(other, 0): return self.sub(other)
        return self
    def lazy_sub_(self, other: int | float | list[int | float] | tuple[int | float]):
        if generic_ne(other, 0): return self.sub_(other)
        return self

    def neg(self): return self.__class__(torch._foreach_neg(self))
    def neg_(self):
        torch._foreach_neg_(self)
        return self

    def mul(self, other: _STOrSTSeq): return self.__class__(torch._foreach_mul(self, other))
    def mul_(self, other: _STOrSTSeq):
        torch._foreach_mul_(self, other)
        return self

    def lazy_mul(self, other: int | float | list[int | float] | tuple[int | float], clone=False):
        if generic_ne(other, 1):
            return self * other
        if clone: return self.clone()
        return self
    def lazy_mul_(self, other: int | float | list[int | float] | tuple[int | float]):
        if generic_ne(other, 1): return self.mul_(other)
        return self

    def div(self, other: _STOrSTSeq) -> Self: return self.__class__(torch._foreach_div(self, other))
    def div_(self, other: _STOrSTSeq):
        torch._foreach_div_(self, other)
        return self

    def lazy_div(self, other: int | float | list[int | float] | tuple[int | float]):
        if generic_ne(other, 1): return self / other
        return self
    def lazy_div_(self, other: int | float | list[int | float] | tuple[int | float]):
        if generic_ne(other, 1): return self.div_(other)
        return self

    def pow(self, exponent: "_Scalar | _STSeq"): return self.__class__(torch._foreach_pow(self, exponent))
    def pow_(self, exponent: "_Scalar | _STSeq"):
        torch._foreach_pow_(self, exponent)
        return self

    def lazy_pow(self, other: int | float | list[int | float] | tuple[int | float]):
        if generic_ne(other, 1): return self.pow(other)
        return self
    def lazy_pow_(self, other: int | float | list[int | float] | tuple[int | float]):
        if generic_ne(other, 1): return self.pow_(other)
        return self

    def rpow(self, input: _Scalar | _TensorSeq): return self.__class__(torch._foreach_pow(input, self))
    def rpow_(self, input: _TensorSeq):
        torch._foreach_pow_(input, self)
        return self

    def square(self): return self.__class__(torch._foreach_pow(self, 2))
    def square_(self):
        torch._foreach_pow_(self, 2)
        return self

    def sqrt(self): return self.__class__(torch._foreach_sqrt(self))
    def sqrt_(self):
        torch._foreach_sqrt_(self)
        return self

    def remainder(self, other: _STOrSTSeq): return self.zipmap(torch.remainder, other)
    def remainder_(self, other: _STOrSTSeq): return self.zipmap_inplace_(_MethodCallerWithArgs('remainder_'), other)

    def floor_divide(self, other: _STOrSTSeq): return self.zipmap(torch.floor_divide, other)
    def floor_divide_(self, other: _STOrSTSeq): return self.zipmap_inplace_(_MethodCallerWithArgs('floor_divide_'), other)

    def reciprocal(self): return self.__class__(torch._foreach_reciprocal(self))
    def reciprocal_(self):
        torch._foreach_reciprocal_(self)
        return self

    def abs(self): return self.__class__(torch._foreach_abs(self))
    def abs_(self):
        torch._foreach_abs_(self)
        return self

    def sign(self): return self.__class__(torch._foreach_sign(self))
    def sign_(self):
        torch._foreach_sign_(self)
        return self

    def exp(self): return self.__class__(torch._foreach_exp(self))
    def exp_(self):
        torch._foreach_exp_(self)
        return self

    def signbit(self): return self.__class__(torch.signbit(i) for i in self)

    def sin(self): return self.__class__(torch._foreach_sin(self))
    def sin_(self):
        torch._foreach_sin_(self)
        return self

    def cos(self): return self.__class__(torch._foreach_cos(self))
    def cos_(self):
        torch._foreach_cos_(self)
        return self

    def tan(self): return self.__class__(torch._foreach_tan(self))
    def tan_(self):
        torch._foreach_tan_(self)
        return self

    def asin(self): return self.__class__(torch._foreach_asin(self))
    def asin_(self):
        torch._foreach_asin_(self)
        return self

    def acos(self): return self.__class__(torch._foreach_acos(self))
    def acos_(self):
        torch._foreach_acos_(self)
        return self

    def atan(self): return self.__class__(torch._foreach_atan(self))
    def atan_(self):
        torch._foreach_atan_(self)
        return self

    def sinh(self): return self.__class__(torch._foreach_sinh(self))
    def sinh_(self):
        torch._foreach_sinh_(self)
        return self

    def cosh(self): return self.__class__(torch._foreach_cosh(self))
    def cosh_(self):
        torch._foreach_cosh_(self)
        return self

    def tanh(self): return self.__class__(torch._foreach_tanh(self))
    def tanh_(self):
        torch._foreach_tanh_(self)
        return self

    def log(self): return self.__class__(torch._foreach_log(self))
    def log_(self):
        torch._foreach_log_(self)
        return self

    def log10(self): return self.__class__(torch._foreach_log10(self))
    def log10_(self):
        torch._foreach_log10_(self)
        return self

    def log2(self): return self.__class__(torch._foreach_log2(self))
    def log2_(self):
        torch._foreach_log2_(self)
        return self

    def log1p(self): return self.__class__(torch._foreach_log1p(self))
    def log1p_(self):
        torch._foreach_log1p_(self)
        return self

    def erf(self): return self.__class__(torch._foreach_erf(self))
    def erf_(self):
        torch._foreach_erf_(self)
        return self

    def erfc(self): return self.__class__(torch._foreach_erfc(self))
    def erfc_(self):
        torch._foreach_erfc_(self)
        return self

    def sigmoid(self): return self.__class__(torch._foreach_sigmoid(self))
    def sigmoid_(self):
        torch._foreach_sigmoid_(self)
        return self

    def _global_fn(self, keepdim, fn, *args, **kwargs):
        """checks that keepdim is False and returns fn(*args, **kwargs)"""
        #if keepdim: raise ValueError('dim = global and keepdim = True')
        return fn(*args, **kwargs)

    def max(self, dim: _Dim = None, keepdim = False) -> Self | Any:
        if dim is None and not keepdim: return self.__class__(torch._foreach_max(self))
        if dim == 'global': return self._global_fn(keepdim, self.global_max)
        if dim is None: dim = ()
        return self.__class__(i.amax(dim=dim, keepdim=keepdim) for i in self)

    def min(self, dim: _Dim = None, keepdim = False) -> Self | Any:
        if dim is None and not keepdim: return self.__class__(torch._foreach_max(self.neg())).neg_()
        if dim == 'global': return self._global_fn(keepdim, self.global_min)
        if dim is None: dim = ()
        return self.__class__(i.amin(dim=dim, keepdim=keepdim) for i in self)

    def norm(self, ord: float, dtype=None):
        return self.__class__(torch._foreach_norm(self, ord, dtype))

    def metric(self, metric: Metrics) -> "TensorList":
        return calculate_metric_list(self, metric)

    def mean(self, dim: _Dim = None, keepdim = False) -> Self | Any:
        if dim == 'global': return self._global_fn(keepdim, self.global_mean)
        return self.__class__(i.mean(dim=dim, keepdim=keepdim) for i in self)

    def sum(self, dim: _Dim = None, keepdim = False) -> Self | Any:
        if dim == 'global': return self._global_fn(keepdim, self.global_sum)
        return self.__class__(i.sum(dim=dim, keepdim=keepdim) for i in self)

    def prod(self, dim = None, keepdim = False): return self.__class__(i.prod(dim=dim, keepdim=keepdim) for i in self)

    def std(self, dim: _Dim = None, unbiased: bool = True, keepdim = False) -> Self | Any:
        if dim == 'global': return self._global_fn(keepdim, self.global_std)
        return self.__class__(i.std(dim=dim, unbiased=unbiased, keepdim=keepdim) for i in self)

    def var(self, dim: _Dim = None, unbiased: bool = True, keepdim = False) -> Self | Any:
        if dim == 'global': return self._global_fn(keepdim, self.global_var)
        return self.__class__(i.var(dim=dim, unbiased=unbiased, keepdim=keepdim) for i in self)

    def median(self, dim=None, keepdim=False):
        """note this doesn't return indices"""
        # median returns tensor or namedtuple (values, indices)
        if dim is None: return self.__class__(i.median() for i in self)
        return self.__class__(i.median(dim=dim, keepdim=keepdim)[0] for i in self)

    def quantile(self, q, dim=None, keepdim=False, *, interpolation='linear',):
        return self.__class__(i.quantile(q=q, dim=dim, keepdim=keepdim, interpolation=interpolation) for i in self)

    def clamp_min(self, other: "_Scalar | _STSeq"): return self.__class__(torch._foreach_clamp_min(self, other))
    def clamp_min_(self, other: "_Scalar | _STSeq"):
        torch._foreach_clamp_min_(self, other)
        return self
    def clamp_max(self, other: "_Scalar | _STSeq"): return self.__class__(torch._foreach_clamp_max(self, other))
    def clamp_max_(self, other: "_Scalar | _STSeq"):
        torch._foreach_clamp_max_(self, other)
        return self

    def clamp(self, min: "_Scalar | _STSeq | None" = None, max: "_Scalar | _STSeq | None" = None):
        l = self
        if min is not None: l = l.clamp_min(min)
        if max is not None: l = l.clamp_max(max)
        return l
    def clamp_(self, min: "_Scalar | _STSeq | None" = None, max: "_Scalar | _STSeq | None" = None):
        if min is not None: self.clamp_min_(min)
        if max is not None: self.clamp_max_(max)
        return self

    def clip(self, min: "_Scalar | _STSeq | None" = None, max: "_Scalar | _STSeq | None" = None): return self.clamp(min,max)
    def clip_(self, min: "_Scalar | _STSeq | None" = None, max: "_Scalar | _STSeq | None" = None): return self.clamp_(min,max)

    def clamp_magnitude(self, min: "_Scalar | _STSeq | None" = None, max: "_Scalar | _STSeq | None" = None):
        return self.abs().clamp_(min, max) * self.sign().add_(0.5).sign_() # this prevents zeros
    def clamp_magnitude_(self, min: "_Scalar | _STSeq | None" = None, max: "_Scalar | _STSeq | None" = None):
        sign = self.sign().add_(0.5).sign_()
        return self.abs_().clamp_(min, max).mul_(sign)


    def floor(self): return self.__class__(torch._foreach_floor(self))
    def floor_(self):
        torch._foreach_floor_(self)
        return self
    def ceil(self): return self.__class__(torch._foreach_ceil(self))
    def ceil_(self):
        torch._foreach_ceil_(self)
        return self
    def round(self): return self.__class__(torch._foreach_round(self))
    def round_(self):
        torch._foreach_round_(self)
        return self

    def zero_(self):
        torch._foreach_zero_(self)
        return self

    def lerp(self, tensors1: _TensorSeq, weight: "_Scalar | _ScalarSeq | _TensorSeq"):
        """linear interpolation of between self and tensors1. `out = self + weight * (tensors1 - self)`."""
        return self.__class__(torch._foreach_lerp(self, tensors1, weight))
    def lerp_(self, tensors1: _TensorSeq, weight: "_Scalar | _ScalarSeq | _TensorSeq"):
        """linear interpolation of between self and tensors1. `out = self + weight * (tensors1 - self)`."""
        torch._foreach_lerp_(self, tensors1, weight)
        return self

    def lerp_compat(self, tensors1: _TensorSeq, weight: "_STOrSTSeq"):
        """`lerp` but support scalar sequence weight on pytorch versions before 2.6

        `out = self + weight * (tensors1 - self)`."""
        return self + weight * (TensorList(tensors1) - self)
    def lerp_compat_(self, tensors1: _TensorSeq, weight: "_STOrSTSeq"):
        """`lerp_` but support scalar sequence weight on previous pytorch versions before 2.6

        `out = self + weight * (tensors1 - self)`."""
        return self.add_(TensorList(tensors1).sub(self).mul_(weight))

    def addcmul(self, tensors1: _TensorSeq, tensor2: _TensorSeq, value: "_Scalar | Sequence[_Scalar] | torch.Tensor" = 1):
        return self.__class__(torch._foreach_addcmul(self, tensors1, tensor2, value))
    def addcmul_(self, tensors1: _TensorSeq, tensor2: _TensorSeq, value: "_Scalar | Sequence[_Scalar] | torch.Tensor" = 1):
        torch._foreach_addcmul_(self, tensors1, tensor2, value)
        return self
    def addcdiv(self, tensors1: _TensorSeq, tensor2: _TensorSeq, value: "_Scalar | Sequence[_Scalar] | torch.Tensor" = 1):
        return self.__class__(torch._foreach_addcdiv(self, tensors1, tensor2, value))
    def addcdiv_(self, tensors1: _TensorSeq, tensor2: _TensorSeq, value: "_Scalar | Sequence[_Scalar] | torch.Tensor" = 1):
        torch._foreach_addcdiv_(self, tensors1, tensor2, value)
        return self

    def uniform_(self, low: "_Scalar | _ScalarSeq" = 0, high: "_Scalar | _ScalarSeq" = 1, generator = None):
        return self.zipmap_args_inplace_(_MethodCallerWithArgs('uniform_'), low, high, generator = generator)

    def maximum(self, other: "_Scalar | _ScalarSeq | _TensorSeq"):
        return self.__class__(torch._foreach_maximum(self, other))
    def maximum_(self, other: "_Scalar | _ScalarSeq | _TensorSeq"): # ruff: noqa F811
        torch._foreach_maximum_(self, other)
        return self

    def minimum(self, other: "_Scalar | _ScalarSeq | _TensorSeq"):
        return self.__class__(torch._foreach_minimum(self, other))
    def minimum_(self, other: "_Scalar | _ScalarSeq | _TensorSeq"):
        torch._foreach_minimum_(self, other)
        return self

    def squeeze(self, dim = None):
        if dim is None: return self.__class__(i.squeeze() for i in self)
        return self.__class__(i.squeeze(dim) for i in self)

    def squeeze_(self, dim = None):
        if dim is None:
            for i in self: i.squeeze_()
        else:
            for i in self: i.squeeze_(dim)
        return self

    def conj(self): return self.__class__(i.conj() for i in self)

    def nan_to_num(self, nan: "float | _ScalarSeq | None" = None, posinf: "float | _ScalarSeq | None" = None, neginf: "float | _ScalarSeq | None" = None):
        return self.zipmap_args(torch.nan_to_num, nan, posinf, neginf)
    def nan_to_num_(self, nan: "float | _ScalarSeq | None" = None, posinf: "float | _ScalarSeq | None" = None, neginf: "float | _ScalarSeq | None" = None):
        return self.zipmap_args_inplace_(torch.nan_to_num_, nan, posinf, neginf)

    def ravel(self): return self.__class__(i.ravel() for i in self)
    def view_flat(self): return self.__class__(i.view(-1) for i in self)

    def any(self): return self.__class__(i.any() for i in self)
    def all(self): return self.__class__(i.all() for i in self)
    def isfinite(self): return self.__class__(i.isfinite() for i in self)

    def fill(self, value: _STOrSTSeq): return self.zipmap(torch.fill, other = value)
    def fill_(self, value: _STOrSTSeq): return self.zipmap_inplace_(torch.fill_, other = value)

    def copysign(self, other):
        return self.__class__(t.copysign(o) for t, o in zip(self, other))
    def copysign_(self, other):
        for t, o in zip(self, other): t.copysign_(o)
        return self

    def graft(self, magnitude: "_TensorSeq", tensorwise=False, ord: Metrics = 2, eps = 1e-6, strength: float | _ScalarSeq = 1):
        if not isinstance(magnitude, TensorList): magnitude = TensorList(magnitude)
        if tensorwise:
            norm_self = self.metric(ord)
            norm_other = magnitude.metric(ord)
        else:
            norm_self = self.global_metric(ord)
            norm_other = magnitude.global_metric(ord)

        if generic_ne(strength, 1): norm_other.lerp_(norm_self, 1-maybe_numberlist(strength)) # pyright:ignore[reportCallIssue,reportArgumentType]

        return self * (norm_other / norm_self.clip_(min=eps))

    def graft_(self, magnitude: "_TensorSeq", tensorwise=False, ord: Metrics = 2, eps = 1e-6, strength: float | _ScalarSeq = 1):
        if not isinstance(magnitude, TensorList): magnitude = TensorList(magnitude)
        if tensorwise:
            norm_self = self.metric(ord)
            norm_other = magnitude.metric(ord)
        else:
            norm_self = self.global_metric(ord)
            norm_other = magnitude.global_metric(ord)

        if generic_ne(strength, 1): norm_other.lerp_(norm_self, 1-maybe_numberlist(strength)) # pyright:ignore[reportCallIssue,reportArgumentType]

        return self.mul_(norm_other / norm_self.clip_(min=eps))

    def _get_rescale_coeffs(self, min:"_Scalar | _ScalarSeq", max:"_Scalar | _ScalarSeq", dim: _Dim, eps):
        self_min = self.min(dim=dim, keepdim=True)
        self_max = self.max(dim=dim, keepdim=True)

        # target range difference (diff)
        min = maybe_numberlist(min)
        max = maybe_numberlist(max)
        diff = max - min
        target_min = min
        source_range = (self_max - self_min).add_(eps)
        a = diff / source_range
        b = target_min - (a * self_min)

        return a, b

    def rescale(self, min: "_Scalar | _ScalarSeq | None", max: "_Scalar | _ScalarSeq | None", dim: _Dim = None, eps=0.):
        """rescales each tensor to (min, max) range"""
        if min is None and max is None: return self
        if max is None:
            assert min is not None
            return self - (self.min(dim=dim, keepdim=True).sub_(min))
        if min is None: return self - (self.max(dim=dim, keepdim=True).sub_(max))

        a,b = self._get_rescale_coeffs(min=min, max=max, dim=dim, eps=eps)
        return (self*a).add_(b)

    def rescale_(self, min: "_Scalar | _ScalarSeq | None", max: "_Scalar | _ScalarSeq | None", dim: _Dim = None, eps=0.):
        """rescales each tensor to (min, max) range"""
        if min is None and max is None: return self
        if max is None:
            assert min is not None
            return self.sub_(self.min(dim=dim, keepdim=True).sub_(min))
        if min is None: return self.sub_(self.max(dim=dim, keepdim=True).sub_(max))

        a,b = self._get_rescale_coeffs(min=min, max=max, dim=dim, eps=eps)
        return (self.mul_(a)).add_(b)

    def rescale_to_01(self, dim: _Dim = None, eps: float = 0):
        """faster method to rescale to (0, 1) range"""
        res = self - self.min(dim = dim, keepdim=True)
        max = res.max(dim = dim, keepdim=True)
        if eps != 0: max.add_(eps)
        return res.div_(max)

    def rescale_to_01_(self, dim: _Dim = None, eps: float = 0):
        """faster method to rescale to (0, 1) range"""
        self.sub_(self.min(dim = dim, keepdim=True))
        max = self.max(dim = dim, keepdim=True)
        if eps != 0: max.add_(eps)
        return self.div_(max)

    def normalize(self, mean: "_Scalar | _ScalarSeq | None", var: "_Scalar | _ScalarSeq | None", dim: _Dim = None): # pylint:disable=redefined-outer-name
        """normalizes to mean and variance"""
        if mean is None and var is None: return self
        if mean is None: return self / self.std(dim = dim, keepdim = True)
        if var is None: return self - self.mean(dim = dim, keepdim = True)
        self_mean = self.mean(dim = dim, keepdim = True)
        self_std = self.std(dim = dim, keepdim = True)

        if isinstance(var, Sequence): var_sqrt = [i**0.5 for i in var]
        else: var_sqrt = var ** 0.5

        return (self - self_mean).div_(self_std).mul_(var_sqrt).add_(mean)

    def normalize_(self, mean: "_Scalar | _ScalarSeq | None", var: "_Scalar | _ScalarSeq | None", dim: _Dim = None): # pylint:disable=redefined-outer-name
        """normalizes to mean and variance"""
        if mean is None and var is None: return self
        if mean is None: return self / self.std(dim = dim, keepdim = True)
        if var is None: return self - self.mean(dim = dim, keepdim = True)
        self_mean = self.mean(dim = dim, keepdim = True)
        self_std = self.std(dim = dim, keepdim = True)

        if isinstance(var, Sequence): var_sqrt = [i**0.5 for i in var]
        else: var_sqrt = var ** 0.5

        return self.sub_(self_mean).div_(self_std).mul_(var_sqrt).add_(mean)

    def znormalize(self, dim: _Dim = None, eps:float = 0):
        """faster method to normalize to 0 mean and 1 variance"""
        std = self.std(dim = dim, keepdim = True)
        if eps!=0: std.add_(eps)
        return (self - self.mean(dim = dim, keepdim=True)).div_(std)

    def znormalize_(self, dim: _Dim = None, eps:float = 0):
        """faster method to normalize to 0 mean and 1 variance"""
        std = self.std(dim = dim, keepdim = True)
        if eps!=0: std.add_(eps)
        return self.sub_(self.mean(dim = dim, keepdim=True)).div_(std)

    def _clip_multiplier(self, min: "_Scalar | _ScalarSeq | None"= None, max: "_Scalar | _ScalarSeq | None" = None, tensorwise: bool = True, ord:Metrics = 2):
        """calculate multipler to clip self norm to min and max"""
        if tensorwise:
            self_norm = self.metric(ord)
            self_norm.masked_fill_(self_norm == 0, 1)

        else:
            self_norm = self.global_metric(ord)
            if self_norm == 0: return 1

        mul = 1
        if min is not None:
            mul_to_min = generic_clamp(maybe_numberlist(min) / self_norm, min=1)
            mul *= mul_to_min

        if max is not None:
            mul_to_max = generic_clamp(maybe_numberlist(max) / self_norm, max=1)
            mul *= mul_to_max

        return mul

    def clip_norm(self, min: "_Scalar | _ScalarSeq | None"= None, max: "_Scalar | _ScalarSeq | None" = None, tensorwise: bool = True, ord:Metrics = 2):
        """clips norm of each tensor to (min, max) range"""
        if min is None and max is None: return self
        return self * self._clip_multiplier(min, max, tensorwise, ord)

    def clip_norm_(self, min: "_Scalar | _ScalarSeq | None"= None, max: "_Scalar | _ScalarSeq | None" = None, tensorwise: bool = True, ord:Metrics = 2):
        """clips norm of each tensor to (min, max) range"""
        if min is None and max is None: return self
        return self.mul_(self._clip_multiplier(min, max, tensorwise, ord))


    def where(self, condition: "torch.Tensor | _TensorSeq", other: _STOrSTSeq):
        """self where condition is true other otherwise"""
        return self.zipmap_args(_MethodCallerWithArgs('where'), condition, other)

    def masked_fill(self, mask: "torch.Tensor | _TensorSeq", fill_value: "_Scalar | _ScalarSeq"):
        """Same as tensor[mask] = value (not in-place), where value must be scalar/scalars"""
        return self.zipmap_args(torch.masked_fill, mask, fill_value)
    def masked_fill_(self, mask: "torch.Tensor | _TensorSeq", fill_value: "_Scalar | _ScalarSeq"):
        """Same as tensor[mask] = value, where value must be scalar/scalars"""
        return self.zipmap_args_inplace_(_MethodCallerWithArgs('masked_fill_'), mask, fill_value)

    def select_set_(self, mask: _TensorSeq, value: _STOrSTSeq):
        """Same as tensor[mask] = value"""
        list_value = value if isinstance(value, (list,tuple)) else [value]*len(self)
        for tensor, m, v in zip(self, mask, list_value):
            tensor[m] = v # pyright: ignore[reportArgumentType]

    def masked_set_(self, mask: _TensorSeq, value: _TensorSeq):
        """Same as tensor[mask] = value[mask]"""
        for tensor, m, v in zip(self, mask, value):
            tensor[m] = v[m]

    def select(self, idx: Any):
        """same as tensor[idx]"""
        if not isinstance(idx, (list,tuple)): return self.__class__(t[idx] for t in self)
        return self.__class__(t[i] for t,i in zip(self, idx))

    def dot(self, other: _TensorSeq):
        return (self * other).global_sum()

    def tensorwise_dot(self, other: _TensorSeq):
        return (self * other).sum()

    def swap_tensors(self, other: _TensorSeq):
        for s, o in zip(self, other):
            torch.utils.swap_tensors(s, o)

    def unbind_channels(self, dim=0):
        """returns a new tensorlist where tensors with 2 or more dimensions are split into slices along 1st dimension"""
        return self.__class__(ch for t in self for ch in (t.unbind(dim) if t.ndim >= 2 else (t,)) )


    def flatiter(self) -> Generator[torch.Tensor]:
        for tensor in self:
            yield from tensor.view(-1)

    # def flatset(self, idx: int, value: Any):
    #     """sets index in flattened view"""
    #     return self.clone().flatset_(idx, value)

    def flat_get(self, idx: int):
        cur = 0
        for tensor in self:
            numel = tensor.numel()
            if idx < cur + numel:
                return tensor.view(-1)[cur-idx]
            cur += numel
        raise IndexError(idx)

    def flat_set_(self, idx: int, value: Any):
        """sets index in flattened view"""
        cur = 0
        for tensor in self:
            numel = tensor.numel()
            if idx < cur + numel:
                tensor.view(-1)[cur-idx] = value
                return self
            cur += numel
        raise IndexError(idx)

    def flat_set_lambda_(self, idx, fn):
        """sets index in flattened view to return of fn(current_value)"""
        cur = 0
        for tensor in self:
            numel = tensor.numel()
            if idx < cur + numel:
                flat_view = tensor.view(-1)
                flat_view[cur-idx] = fn(flat_view[cur-idx])
                return self
            cur += numel
        raise IndexError(idx)

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"


def stack(tensorlists: Iterable[TensorList], dim = 0):
    """Returns a tensorlist with the same elements as the input tensorlists, but stacked along the specified dimension."""
    return TensorList(torch.stack(i, dim = dim) for i in zip(*tensorlists))

def mean(tensorlists: Iterable[TensorList]):
    """Returns a tensorlist which is the mean of given tensorlists."""
    res = TensorList()
    for tensors in zip(*tensorlists):
        res.append(torch.stack(tensors).mean(0))
    return res

def median(tensorlists: Iterable[TensorList]):
    """Returns a tensorlist which is the median of given tensorlists."""
    res = TensorList()
    for tensors in zip(*tensorlists):
        res.append(torch.stack(tensors).median(0)[0])
    return res

def quantile(tensorlists: Iterable[TensorList], q, interpolation = 'linear'):
    """Returns a tensorlist which is the median of given tensorlists."""
    res = TensorList()
    for tensors in zip(*tensorlists):
        res.append(torch.stack(tensors).quantile(q=q, dim=0, interpolation=interpolation))
    return res

def sum(tensorlists: Iterable[TensorList]):
    """Returns a tensorlist which is the sum of given tensorlists."""
    res = TensorList()
    for tensors in zip(*tensorlists):
        res.append(torch.stack(tensors).sum(0))
    return res

def where(condition: TensorList, input: _STOrSTSeq, other: _STOrSTSeq):
    """Where but for a tensorlist."""
    args = [i if isinstance(i, (list, tuple)) else [i]*len(condition) for i in (input, other)]
    return condition.__class__(torch.where(*z) for z in zip(condition, *args))

def generic_clamp(x: Any, min=None,max=None) -> Any:
    if isinstance(x, (torch.Tensor, TensorList)): return x.clamp(min,max)
    if isinstance(x, (list, tuple)): return x.__class__(generic_clamp(i,min,max) for i in x)
    if x < min: return min
    if x > max: return max
    return x

def generic_numel(x: torch.Tensor | TensorList) -> int:
    if isinstance(x, torch.Tensor): return x.numel()
    return x.global_numel()


def generic_finfo(x: torch.Tensor | TensorList) -> torch.finfo:
    if isinstance(x, torch.Tensor): return torch.finfo(x.dtype)
    return torch.finfo(x[0].dtype)

def generic_finfo_eps(x: torch.Tensor | TensorList) -> float:
    if isinstance(x, torch.Tensor): return torch.finfo(x.dtype).eps
    return torch.finfo(x[0].dtype).eps

def generic_finfo_tiny(x: torch.Tensor | TensorList) -> float:
    if isinstance(x, torch.Tensor): return torch.finfo(x.dtype).tiny
    return torch.finfo(x[0].dtype).tiny

@overload
def generic_zeros_like(x: torch.Tensor) -> torch.Tensor: ...
@overload
def generic_zeros_like(x: TensorList) -> TensorList: ...
def generic_zeros_like(x: torch.Tensor | TensorList):
    if isinstance(x, torch.Tensor): return torch.zeros_like(x)
    return x.zeros_like()

def generic_vector_norm(x: torch.Tensor | TensorList, ord=2) -> torch.Tensor:
    if isinstance(x, torch.Tensor): return torch.linalg.vector_norm(x, ord=ord) # pylint:disable=not-callable
    return x.global_vector_norm(ord)

def generic_metric(x: torch.Tensor | TensorList, metric: Metrics) -> torch.Tensor:
    return evaluate_metric(x, metric)

@overload
def generic_randn_like(x: torch.Tensor) -> torch.Tensor: ...
@overload
def generic_randn_like(x: TensorList) -> TensorList: ...
def generic_randn_like(x: torch.Tensor | TensorList):
    if isinstance(x, torch.Tensor): return torch.randn_like(x)
    return x.randn_like()


def generic_sum(x: torch.Tensor | TensorList) -> torch.Tensor:
    if isinstance(x, torch.Tensor): return x.sum()
    return x.global_sum()

def generic_max(x: torch.Tensor | TensorList) -> torch.Tensor:
    if isinstance(x, torch.Tensor): return x.max()
    return x.global_max()
