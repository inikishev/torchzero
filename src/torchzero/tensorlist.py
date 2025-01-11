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
from collections.abc import Callable, Sequence, Iterable, Generator
import math
import operator
from typing import TypeAlias, Any, Literal, TypedDict, Self, Unpack

import torch

_Scalar = int | float | bool | complex
_AnyTensor = torch.Tensor | torch.nn.Parameter
_TensorSequence = list[_AnyTensor] | tuple[_AnyTensor, ...]
_ScalarSequence: TypeAlias = "list[_Scalar] | tuple[_Scalar] | TensorList"
_STSequence: TypeAlias = "_TensorSequence | _ScalarSequence"
_STOrSTSequence: TypeAlias = "_Scalar | torch.Tensor | torch.nn.Parameter | _STSequence"

Distributions = Literal['normal', 'uniform', 'sphere', 'rademacher']
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


def maximum_(input:torch.Tensor, other: torch.Tensor):
    """in-place maximum"""
    return torch.maximum(input, other, out = input)

def where_(input: torch.Tensor, condition: torch.Tensor, other: torch.Tensor):
    """in-place where"""
    return torch.where(condition, input, other, out = input)

# tensorlist must subclass list
# UserList doesn't work with _foreach_xxx
class TensorList(list[torch.Tensor | Any]):
    @classmethod
    def complex(cls, real: _TensorSequence, imag: _TensorSequence):
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

    def to_real_views(self):
        """Turns all complex tensors into real views, ignoring non-complex tensors, and sets an attribute `_tl_is_complex` to True or False,
        which `from_real_views` method can use to convert real views back into complex tensors"""
        tl = TensorList()
        for p in self:
            if torch.is_complex(p):
                p._tl_is_complex = True # type:ignore
                tl.append(torch.view_as_real(p))
            else:
                p._tl_is_complex = False # type:ignore
                tl.append(p)
        return tl

    def from_real_views(self):
        """undoes `to_real_views`."""
        return self.__class__(torch.view_as_complex(p) if p._tl_is_complex else p for p in self) # type:ignore

    def get_existing_grads(self):
        """Returns all gradients that are not None."""
        return self.__class__(i.grad for i in self if i is not None)

    def with_requires_grad(self, requires_grad = True):
        """Returns all tensors with requires_grad set to the given value."""
        return self.__class__(i for i in self if i.requires_grad == requires_grad)

    def ensure_grad_(self):
        """For each element, if grad is None and it requires grad, sets grad to zeroes."""
        for i in self:
            if i.requires_grad and i.grad is None: i.grad = torch.zeros_like(i)
        return self

    def accumulate_grad_(self, grads: _TensorSequence):
        """Creates grad if it is None, otherwise adds to existing grad."""
        for i, g in zip(self, grads):
            if i.grad is None: i.grad = g
            else: i.grad.add_(g)
        return self

    def set_grad_(self, grads: _TensorSequence):
        """Sets grad to the given sequence, overwrites grad that already exists."""
        for i, g in zip(self, grads): i.grad = g
        return self

    def zero_grad_(self, set_to_none = True):
        """Set all grads to None or zeroes."""
        if set_to_none:
            for p in self: p.grad = None
        else:
            self.get_existing_grads().zero_()
        return self

    def __add__(self, other: _STOrSTSequence) -> Self: return self.add(other) # type:ignore
    def __radd__(self, other: _STOrSTSequence) -> Self: return self.add(other)
    def __iadd__(self, other: _STOrSTSequence) -> Self: return self.add_(other) # type:ignore

    def __sub__(self, other: "_Scalar | _STSequence") -> Self: return self.sub(other)
    def __rsub__(self, other: "_Scalar | _STSequence") -> Self: return self.sub(other).neg_()
    def __isub__(self, other: "_Scalar | _STSequence") -> Self: return self.sub_(other)

    def __mul__(self, other: _STOrSTSequence) -> Self: return self.mul(other) # type:ignore
    def __rmul__(self, other: _STOrSTSequence) -> Self: return self.mul(other) # type:ignore
    def __imul__(self, other: _STOrSTSequence) -> Self: return self.mul_(other) # type:ignore

    def __truediv__(self, other: "_Scalar | _STSequence") -> Self: return self.div(other)
    def __rtruediv__(self, other: "_Scalar | _STSequence") -> Self: return other * self.reciprocal() # type:ignore
    def __itruediv__(self, other: "_Scalar | _STSequence") -> Self: return self.div_(other)

    def __floordiv__(self, other: _STOrSTSequence): return self.floor_divide(other)
    #def __rfloordiv__(self, other: "TensorList"): return other.floor_divide(self)
    def __ifloordiv__(self, other: _STOrSTSequence): return self.floor_divide_(other)

    def __mod__(self, other: _STOrSTSequence): return self.remainder(other)
    #def __rmod__(self, other: STOrSTSequence): return self.remainder(other)
    def __imod__(self, other: _STOrSTSequence):return self.remainder_(other)

    def __pow__(self, other: "_Scalar | _STSequence"): return self.pow(other)
    def __rpow__(self, other: "_Scalar | _TensorSequence"): return self.rpow(other)
    def __ipow__(self, other: "_Scalar | _STSequence"): return self.pow_(other)

    def __neg__(self): return self.neg()

    def __eq__(self, other: _STOrSTSequence): return self.eq(other) # type:ignore
    def __ne__(self, other: _STOrSTSequence): return self.ne(other) # type:ignore
    def __lt__(self, other: _STOrSTSequence): return self.lt(other) # type:ignore
    def __le__(self, other: _STOrSTSequence): return self.le(other) # type:ignore
    def __gt__(self, other: _STOrSTSequence): return self.gt(other) # type:ignore
    def __ge__(self, other: _STOrSTSequence): return self.ge(other) # type:ignore

    def __invert__(self): return self.logical_not()

    def __and__(self, other: torch.Tensor | _TensorSequence): return self.logical_and(other)
    def __iand__(self, other: torch.Tensor | _TensorSequence): return self.logical_and_(other)
    def __or__(self, other: torch.Tensor | _TensorSequence): return self.logical_or(other)
    def __ior__(self, other: torch.Tensor | _TensorSequence): return self.logical_or_(other)
    def __xor__(self, other: torch.Tensor | _TensorSequence): return self.logical_xor(other)
    def __ixor__(self, other: torch.Tensor | _TensorSequence): return self.logical_xor_(other)

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

    def zipmap(self, fn: Callable, other: Any | list | tuple, *args, **kwargs):
        """If `other` is list/tuple, applies `fn` to this TensorList zipped with `other`.
        Otherwise applies `fn` to this TensorList and `other`.
        Returns a new TensorList with return values of the callable."""
        if isinstance(other, (list, tuple)): return self.__class__(fn(i, j, *args, **kwargs) for i, j in zip(self, other))
        return self.__class__(fn(i, other, *args, **kwargs) for i in self)

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
    def detach_(self): return self.__class__(i.detach_() for i in self)

    # apparently I can't use float for typing if I call a method "float"
    def as_float(self): return self.__class__(i.float() for i in self)
    def as_bool(self): return self.__class__(i.bool() for i in self)
    def as_int(self): return self.__class__(i.int() for i in self)

    def copy_(self, src: _TensorSequence, non_blocking = False):
        """Copies the elements from src tensors into self tensors."""
        torch._foreach_copy_(self, src, non_blocking=non_blocking)
    def set_(self, storage: Iterable[torch.Tensor | torch.types.Storage]):
        """Sets elements of this TensorList to the values of a list of tensors."""
        for i, j in zip(self, storage): i.set_(j) # type:ignore
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
            el.copy_(vec[cur:cur + numel].reshape(el.shape)) # type:ignore
            cur += numel
        return self

    def from_vec(self, vec:torch.Tensor):
        """Creates a new TensorList from this TensorList but with values from a 1D tensor.
        The length of the tensor must be equal to the total number of elements in this TensorList."""
        res = []
        cur = 0
        for el in self:
            numel = el.numel()
            res.append(vec[cur:cur + numel].reshape(el.shape)) # type:ignore
            cur += numel
        return TensorList(res)

    def total_min(self) -> torch.Tensor:
        return torch.min(self.to_vec())
    def total_max(self) -> torch.Tensor:
        return torch.max(self.to_vec())
    def total_mean(self) -> torch.Tensor:
        return torch.mean(self.to_vec())
    def total_sum(self) -> torch.Tensor:
        return torch.sum(self.to_vec())
    def total_vector_norm(self, ord:float = 2) -> torch.Tensor:
        return torch.linalg.vector_norm(self.to_vec(), ord = ord) # pylint:disable = not-callable
    def total_any(self):
        return self.to_vec().any()
    def total_all(self):
        return self.to_vec().all()
    def total_numel(self):
        return builtins.sum(self.numel())

    def empty_like(self, **kwargs: Unpack[_NewTensorKwargs]): return self.__class__(torch.empty_like(i, **kwargs) for i in self)
    def zeros_like(self, **kwargs: Unpack[_NewTensorKwargs]): return self.__class__(torch.zeros_like(i, **kwargs) for i in self)
    def ones_like(self, **kwargs: Unpack[_NewTensorKwargs]): return self.__class__(torch.ones_like(i, **kwargs) for i in self)
    def full_like(self, fill_value: "_Scalar | _ScalarSequence", **kwargs: Unpack[_NewTensorKwargs]):
        #return self.__class__(torch.full_like(i, fill_value=fill_value, **kwargs) for i in self)
        return self.zipmap(torch.full_like, other=fill_value, **kwargs)

    def rand_like(self, **kwargs: Unpack[_NewTensorKwargs]): return self.__class__(torch.rand_like(i, **kwargs) for i in self)
    def randn_like(self, **kwargs: Unpack[_NewTensorKwargs]): return self.__class__(torch.randn_like(i, **kwargs) for i in self)

    def randint_like(self, low: "_Scalar | _ScalarSequence", high: "_Scalar | _ScalarSequence", **kwargs: Unpack[_NewTensorKwargs]):
        return self.zipmap_args(torch.randint_like, low, high, **kwargs)
    def uniform_like(self, low: "_Scalar | _ScalarSequence" = 0, high: "_Scalar | _ScalarSequence" = 1, generator=None, **kwargs: Unpack[_NewTensorKwargs]):
        res = self.empty_like(**kwargs)
        res.uniform_(low, high, generator=generator)
        return res
    def sphere_like(self, radius: "_Scalar | _ScalarSequence", **kwargs: Unpack[_NewTensorKwargs]) -> Self:
        r = self.randn_like(**kwargs)
        return (r * radius) / r.total_vector_norm() # type:ignore
    def bernoulli(self, generator = None):
        return self.__class__(torch.bernoulli(i, generator=generator) for i in self)
    def bernoulli_like(self, p: "_Scalar | _ScalarSequence" = 0.5, generator = None, **kwargs: Unpack[_NewTensorKwargs]):
        """p is probability of a 1, other values will be 0."""
        return self.__class__(torch.bernoulli(i, generator = generator) for i in self.full_like(p, **kwargs))
    def rademacher_like(self, p: "_Scalar | _ScalarSequence" = 0.5, generator = None, **kwargs: Unpack[_NewTensorKwargs]):
        """p is probability of a 1, other values will be -1."""
        return self.bernoulli_like(p, generator=generator, **kwargs) * 2 - 1

    def sample_like(self, eps: "_Scalar | _ScalarSequence" = 1, distribution: Distributions = 'normal', generator=None, **kwargs: Unpack[_NewTensorKwargs]):
        """Sample around 0."""
        if distribution == 'normal': return self.randn_like(**kwargs) * eps # TODO: generator
        if distribution == 'uniform':
            if isinstance(eps, (list,tuple)):
                return self.uniform_like([-i/2 for i in eps], [i/2 for i in eps], generator=generator, **kwargs) # type:ignore
            return self.uniform_like(-eps/2, eps/2, generator=generator, **kwargs)
        if distribution == 'sphere': return self.sphere_like(eps, **kwargs)
        if distribution == 'rademacher': return self.rademacher_like(generator=generator, **kwargs) * eps
        raise ValueError(f'Unknow distribution {distribution}')

    def eq(self, other: _STOrSTSequence): return self.zipmap(torch.eq, other)
    def eq_(self, other: _STOrSTSequence): return self.zipmap_inplace_(_MethodCallerWithArgs('eq_'), other)
    def ne(self, other: _STOrSTSequence): return self.zipmap(torch.ne, other)
    def ne_(self, other: _STOrSTSequence): return self.zipmap_inplace_(_MethodCallerWithArgs('ne_'), other)
    def lt(self, other: _STOrSTSequence): return self.zipmap(torch.lt, other)
    def lt_(self, other: _STOrSTSequence): return self.zipmap_inplace_(_MethodCallerWithArgs('lt_'), other)
    def le(self, other: _STOrSTSequence): return self.zipmap(torch.le, other)
    def le_(self, other: _STOrSTSequence): return self.zipmap_inplace_(_MethodCallerWithArgs('le_'), other)
    def gt(self, other: _STOrSTSequence): return self.zipmap(torch.gt, other)
    def gt_(self, other: _STOrSTSequence): return self.zipmap_inplace_(_MethodCallerWithArgs('gt_'), other)
    def ge(self, other: _STOrSTSequence): return self.zipmap(torch.ge, other)
    def ge_(self, other: _STOrSTSequence): return self.zipmap_inplace_(_MethodCallerWithArgs('ge_'), other)

    def logical_and(self, other: torch.Tensor | _TensorSequence): return self.zipmap(torch.logical_and, other)
    def logical_and_(self, other: torch.Tensor | _TensorSequence): return self.zipmap_inplace_(_MethodCallerWithArgs('logical_and_'), other)
    def logical_or(self, other: torch.Tensor | _TensorSequence): return self.zipmap(torch.logical_or, other)
    def logical_or_(self, other: torch.Tensor | _TensorSequence): return self.zipmap_inplace_(_MethodCallerWithArgs('logical_or_'), other)
    def logical_xor(self, other: torch.Tensor | _TensorSequence): return self.zipmap(torch.logical_xor, other)
    def logical_xor_(self, other: torch.Tensor | _TensorSequence): return self.zipmap_inplace_(_MethodCallerWithArgs('logical_xor_'), other)

    def logical_not(self): return self.__class__(torch.logical_not(i) for i in self)
    def logical_not_(self):
        for i in self: i.logical_not_()
        return self

    def equal(self, other: torch.Tensor | _TensorSequence):
        """returns TensorList of boolean values, True if two tensors have the same size and elements, False otherwise."""
        return self.zipmap(torch.equal, other)

    def add(self, other: _STOrSTSequence, alpha: _Scalar = 1):
        if alpha == 1: return self.__class__(torch._foreach_add(self, other))
        return self.__class__(torch._foreach_add(self, other, alpha = alpha)) # type:ignore
    def add_(self, other: _STOrSTSequence, alpha: _Scalar = 1):
        if alpha == 1: torch._foreach_add_(self, other)
        else: torch._foreach_add_(self, other, alpha = alpha) # type:ignore
        return self


    def sub(self, other: "_Scalar | _STSequence", alpha: _Scalar = 1):
        if alpha == 1: return self.__class__(torch._foreach_sub(self, other))
        return self.__class__(torch._foreach_sub(self, other, alpha = alpha)) # type:ignore
    def sub_(self, other: "_Scalar | _STSequence", alpha: _Scalar = 1):
        if alpha == 1: torch._foreach_sub_(self, other)
        else: torch._foreach_sub_(self, other, alpha = alpha) # type:ignore
        return self

    def neg(self): return self.__class__(torch._foreach_neg(self))
    def neg_(self):
        torch._foreach_neg_(self)
        return self

    def mul(self, other: _STOrSTSequence): return self.__class__(torch._foreach_mul(self, other))
    def mul_(self, other: _STOrSTSequence):
        torch._foreach_mul_(self, other)
        return self

    def div(self, other: _STOrSTSequence) -> Self: return self.__class__(torch._foreach_div(self, other))
    def div_(self, other: _STOrSTSequence):
        torch._foreach_div_(self, other)
        return self

    def pow(self, exponent: "_Scalar | _STSequence"): return self.__class__(torch._foreach_pow(self, exponent))
    def pow_(self, exponent: "_Scalar | _STSequence"):
        torch._foreach_pow_(self, exponent)
        return self

    def rpow(self, input: _Scalar | _TensorSequence): return self.__class__(torch._foreach_pow(input, self))
    def rpow_(self, input: _TensorSequence):
        torch._foreach_pow_(input, self)
        return self

    def sqrt(self): return self.__class__(torch._foreach_sqrt(self))
    def sqrt_(self):
        torch._foreach_sqrt_(self)
        return self

    def remainder(self, other: _STOrSTSequence): return self.zipmap(torch.remainder, other)
    def remainder_(self, other: _STOrSTSequence): return self.zipmap_inplace_(_MethodCallerWithArgs('remainder_'), other)

    def floor_divide(self, other: _STOrSTSequence): return self.zipmap(torch.floor_divide, other)
    def floor_divide_(self, other: _STOrSTSequence): return self.zipmap_inplace_(_MethodCallerWithArgs('floor_divide_'), other)

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

    def max(self, dim = None, keepdim = False):
        if dim is None and not keepdim: return self.__class__(torch._foreach_max(self))
        return self.__class__(i.max(dim=dim, keepdim=keepdim) for i in self)

    def min(self, dim = None, keepdim = False):
        if dim is None and not keepdim: return self.__class__(torch._foreach_max(self.neg())).neg()
        return self.__class__(i.min(dim=dim, keepdim=keepdim) for i in self)

    def norm(self, ord: _Scalar, dtype=None):
        return self.__class__(torch._foreach_norm(self, ord, dtype))

    def mean(self, dim = None, keepdim = False): return self.__class__(i.mean(dim=dim, keepdim=keepdim) for i in self)
    def sum(self, dim = None, keepdim = False): return self.__class__(i.sum(dim=dim, keepdim=keepdim) for i in self)
    def prod(self, dim = None, keepdim = False): return self.__class__(i.prod(dim=dim, keepdim=keepdim) for i in self)

    def clamp_min(self, other: "_Scalar | _STSequence"): return self.__class__(torch._foreach_clamp_min(self, other))
    def clamp_min_(self, other: "_Scalar | _STSequence"):
        torch._foreach_clamp_min_(self, other)
        return self
    def clamp_max(self, other: "_Scalar | _STSequence"): return self.__class__(torch._foreach_clamp_max(self, other))
    def clamp_max_(self, other: "_Scalar | _STSequence"):
        torch._foreach_clamp_max_(self, other)
        return self

    def clamp(self, min: "_Scalar | _STSequence | None" = None, max: "_Scalar | _STSequence | None" = None):
        l = self
        if min is not None: l = l.clamp_min(min)
        if max is not None: l = l.clamp_max(max)
        return l
    def clamp_(self, min: "_Scalar | _STSequence | None" = None, max: "_Scalar | _STSequence | None" = None):
        if min is not None: self.clamp_min_(min)
        if max is not None: self.clamp_max_(max)
        return self

    def clamp_magnitude(self, min: "_Scalar | _STSequence | None" = None, max: "_Scalar | _STSequence | None" = None):
        return self.abs().clamp_(min, max) * self.sign().add_(0.5).sign_() # this prevents zeros
    def clamp_magnitude_(self, min: "_Scalar | _STSequence | None" = None, max: "_Scalar | _STSequence | None" = None):
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
        torch._foreach_round(self)
        return self

    def zero_(self):
        torch._foreach_zero_(self)
        return self

    def lerp(self, tensors1: _TensorSequence, weight: "_Scalar | _TensorSequence"):
        """linear interpolation of between self and tensors1. `out = self + weight * (tensors1 - self)`."""
        return self.__class__(torch._foreach_lerp(self, tensors1, weight))
    def lerp_(self, tensors1: _TensorSequence, weight: "_Scalar | _TensorSequence"):
        """linear interpolation of between self and tensors1. `out = self + weight * (tensors1 - self)`."""
        torch._foreach_lerp_(self, tensors1, weight)
        return self

    def lerp_compat(self, tensors1: _TensorSequence, weight: "_STOrSTSequence"):
        """`lerp` but supports python number sequence as weight and implemented through other operations

        `out = self + weight * (tensors1 - self)`."""
        return self + weight * (TensorList(tensors1) - self)
    def lerp_compat_(self, tensors1: _TensorSequence, weight: "_STOrSTSequence"):
        """`lerp_` but supports python number sequence as weight and implemented through other operations

        `out = self + weight * (tensors1 - self)`."""
        return self.add_(TensorList(tensors1).sub(self).mul_(weight))

    def addcmul(self, tensors1: _TensorSequence, tensor2: _TensorSequence, value: "_Scalar | Sequence[_Scalar] | torch.Tensor" = 1):
        return self.__class__(torch._foreach_addcmul(self, tensors1, tensor2, value))
    def addcmul_(self, tensors1: _TensorSequence, tensor2: _TensorSequence, value: "_Scalar | Sequence[_Scalar] | torch.Tensor" = 1):
        torch._foreach_addcmul_(self, tensors1, tensor2, value)
        return self
    def addcdiv(self, tensors1: _TensorSequence, tensor2: _TensorSequence, value: "_Scalar | Sequence[_Scalar] | torch.Tensor" = 1):
        return self.__class__(torch._foreach_addcdiv(self, tensors1, tensor2, value))
    def addcdiv_(self, tensors1: _TensorSequence, tensor2: _TensorSequence, value: "_Scalar | Sequence[_Scalar] | torch.Tensor" = 1):
        torch._foreach_addcdiv_(self, tensors1, tensor2, value)
        return self

    def uniform_(self, low: "_Scalar | _ScalarSequence" = 0, high: "_Scalar | _ScalarSequence" = 1, generator = None):
        return self.zipmap_args_inplace_(_MethodCallerWithArgs('uniform_'), low, high, generator = generator)

    def maximum(self, other: torch.Tensor | _TensorSequence): return self.zipmap(torch.maximum, other = other)
    def maximum_(self, other: torch.Tensor | _TensorSequence): return self.zipmap_inplace_(maximum_, other = other)

    def squeeze(self, dim = None):
        return self.__class__(i.squeeze(dim) for i in self)

    def squeeze_(self, dim = None):
        for i in self: i.squeeze_(dim)
        return self

    def conj(self): return self.__class__(i.conj() for i in self)

    def nan_to_num_(self,nan: float | None = None,posinf: float | None = None,neginf: float | None = None):
        for i in self: torch.nan_to_num_(i, nan = nan, posinf = posinf, neginf = neginf)
        return self

    def ravel(self): return self.__class__(i.ravel() for i in self)

    def any(self): return self.__class__(i.any() for i in self)
    def all(self): return self.__class__(i.all() for i in self)
    def isfinite(self): return self.__class__(i.isfinite() for i in self)

    def fill(self, value: _STOrSTSequence): return self.zipmap(torch.fill, other = value)
    def fill_(self, value: _STOrSTSequence): return self.zipmap_inplace_(torch.fill_, other = value)

    def where(self, condition: "torch.Tensor | _TensorSequence", other: _STOrSTSequence):
        """self where condition is true other otherwise"""
        return self.zipmap_args(_MethodCallerWithArgs('where'), condition, other)
    def where_(self, condition: "torch.Tensor | _TensorSequence", other: "torch.Tensor | _TensorSequence"):
        """self where condition is true other otherwise"""
        return self.zipmap_args_inplace_(where_, condition, other)

    def masked_fill(self, mask: "torch.Tensor | _TensorSequence", fill_value: "_Scalar | _ScalarSequence"):
        """Same as tensor[mask] = value (not in-place), where value must be scalar/scalars"""
        return self.zipmap_args(torch.masked_fill, mask, fill_value)
    def masked_fill_(self, mask: "torch.Tensor | _TensorSequence", fill_value: "_Scalar | _ScalarSequence"):
        """Same as tensor[mask] = value, where value must be scalar/scalars"""
        return self.zipmap_args_inplace_(_MethodCallerWithArgs('masked_fill_'), mask, fill_value)

    def select_set_(self, mask: _TensorSequence, value: _STOrSTSequence):
        """Same as tensor[mask] = value"""
        if not isinstance(value, (list,tuple)): value = [value]*len(self) # type:ignore
        for tensor, m, v in zip(self, mask, value): # type:ignore
            print(tensor, m, v)
            tensor[m] = v

    def masked_set_(self, mask: _TensorSequence, value: _TensorSequence):
        """Same as tensor[mask] = value[mask]"""
        for tensor, m, v in zip(self, mask, value):
            tensor[m] = v[m]

    def select(self, idx: Any):
        """same as tensor[idx]"""
        if not isinstance(idx, (list,tuple)): return self.__class__(t[idx] for t in self)
        return self.__class__(t[i] for t,i in zip(self, idx))

    def swap_tensors(self, other: _TensorSequence):
        for s, o in zip(self, other):
            torch.utils.swap_tensors(s, o)

    def unbind_channels(self, dim=0):
        """returns a new tensorlist where tensors with 2 or more dimensions are split into slices along 1st dimension"""
        return self.__class__(ch for t in self for ch in (t.unbind(dim) if t.ndim >= 2 else (t,)) )

    def flatiter(self) -> Generator[torch.Tensor]:
        for tensor in self:
            yield from tensor.view(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"


def _alpha_add(x, other, alpha):
    return x + other * alpha

class NumberList(TensorList):
    """TensorList subclass for python numbers.
    Note that this only supports basic arithmetic operations that are overloaded.

    Can't use a numpy array because _foreach methods do not work with it."""
    def _set_to_method_result(self, method: str, *args, **kwargs):
        """Sets each element of the tensorlist to the result of calling the specified method on the corresponding element.
        This is used to support/mimic in-place operations."""
        res = getattr(self, method)(*args, **kwargs)
        for i,v in enumerate(res): self[i] = v
        return self

    def add(self, other: _STOrSTSequence, alpha: _Scalar = 1):
        if alpha == 1: return self.zipmap(operator.add, other=other)
        return self.zipmap(_alpha_add, other=other, alpha = alpha)
    def add_(self, other: _STOrSTSequence, alpha: _Scalar = 1):
        return self._set_to_method_result('add', other, alpha = alpha)

    def sub(self, other: "_Scalar | _STSequence", alpha: _Scalar = 1):
        if alpha == 1: return self.zipmap(operator.sub, other=other)
        return self.zipmap(_alpha_add, other=other, alpha = -alpha)

    def sub_(self, other: "_Scalar | _STSequence", alpha: _Scalar = 1):
        return self._set_to_method_result('sub', other, alpha = alpha)

    def neg(self): return self.__class__(-i for i in self)
    def neg_(self): return self._set_to_method_result('neg')

    def mul(self, other: _STOrSTSequence): return self.zipmap(operator.mul, other=other)
    def mul_(self, other: _STOrSTSequence): return self._set_to_method_result('mul', other)

    def div(self, other: _STOrSTSequence) -> Self: return self.zipmap(operator.truediv, other=other)
    def div_(self, other: _STOrSTSequence): return self._set_to_method_result('div', other)

    def pow(self, exponent: "_Scalar | _STSequence"): return self.zipmap(math.pow, other=exponent)
    def pow_(self, exponent: "_Scalar | _STSequence"): return self._set_to_method_result('pow_', exponent)

    def __rtruediv__(self, other: "_Scalar | _STSequence"):
        # overriding because TensorList implements this through reciprocal
        if isinstance(other, (tuple,list)): return self.__class__(o / i for o, i in zip(self, other))
        return self.__class__(other / i for i in self)

def stack(tensorlists: Iterable[TensorList], dim = 0):
    """Returns a tensorlist with the same elements as the input tensorlists, but stacked along the specified dimension."""
    return TensorList(torch.stack(i, dim = dim) for i in zip(*tensorlists))

def mean(tensorlists: Iterable[TensorList]):
    """Returns a tensorlist which is the mean of given tensorlists."""
    return stack(tensorlists).mean(0)

def sum(tensorlists: Iterable[TensorList]):
    """Returns a tensorlist which is the sum of given tensorlists."""
    return stack(tensorlists).sum(0)


def where(condition: TensorList, input: _STOrSTSequence, other: _STOrSTSequence):
    """Where but for a tensorlist."""
    args = [i if isinstance(i, (list, tuple)) else [i]*len(condition) for i in (input, other)]
    return condition.__class__(torch.where(*z) for z in zip(condition, *args))
