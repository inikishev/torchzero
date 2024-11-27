r"""
TensorList is a data type that can be used to manipulate a sequence of tensors such as model parameters,
with the same methods that you use for normal tensors, plus some additional convenience features.
Whenever possible, we use _foreach methods and other tricks to speed up computation.

TensorList is similar to TensorDict (https://github.com/pytorch/tensordict).
If you want to get the most performance out of a collection of tensors, use TensorDict and lock it.
However I found that *creating* a TensorDict is quite slow. In fact it negates the benefits of using it
in an optimizer when you have to create one from parameters on each step. The solution could be to create
it once beforehand, but then you won't be able to easily support parameter groups and per-parameter states.

This is where a TensorList is perfect because creating it has practically no overhead.
"""
import builtins
import collections.abc as A
import math
import operator
import typing as T

import torch

Scalar = int | float | bool | complex
AnyTensor = torch.Tensor | torch.nn.Parameter
TensorSequence = list[AnyTensor] | tuple[AnyTensor, ...]
ScalarSequence: T.TypeAlias = "list[Scalar] | tuple[Scalar] | TensorList"
STSequence: T.TypeAlias = "TensorSequence | ScalarSequence"
STOrSTSequence: T.TypeAlias = "Scalar | torch.Tensor | torch.nn.Parameter | STSequence"

Distributions = T.Literal['normal', 'uniform', 'sphere', 'rademacher']
class _NewTensorKwargs(T.TypedDict, total = False):
    memory_format: T.Any
    dtype: T.Any
    layout: T.Any
    device: T.Any
    pin_memory: bool
    requires_grad: bool

# _foreach_methods = {attr.replace('_foreach_', ''):getattr(torch, attr) for attr in dir(torch) if attr.startswith('_foreach_')}

class MethodCallerWithArgs:
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

# tensorlist must subclass list
# UserList doesn't work with _foreach_xxx
class TensorList(list[torch.Tensor | T.Any]):
    @classmethod
    def complex(cls, real: TensorSequence, imag: TensorSequence):
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

    def accumulate_grad_(self, grads: TensorSequence):
        """Creates grad if it is None, otherwise adds to existing grad."""
        for i, g in zip(self, grads):
            if i.grad is None: i.grad = g
            else: i.grad.add_(g)
        return self

    def set_grad_(self, grads: TensorSequence):
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

    def __add__(self, other: STOrSTSequence): return self.add(other) # type:ignore
    def __radd__(self, other: STOrSTSequence): return self.add(other)
    def __iadd__(self, other: STOrSTSequence): # type:ignore
        self.add_(other)
        return self

    def __sub__(self, other: "Scalar | STSequence"): return self.sub(other)
    def __rsub__(self, other: "Scalar | STSequence"): return - self.sub(other)
    def __isub__(self, other: "Scalar | STSequence"):
        self.sub_(other)
        return self

    def __mul__(self, other: STOrSTSequence): return self.mul(other) # type:ignore
    def __rmul__(self, other: STOrSTSequence): return self.mul(other) # type:ignore
    def __imul__(self, other: STOrSTSequence): # type:ignore
        self.mul_(other)
        return self

    def __truediv__(self, other: "Scalar | STSequence") -> T.Self: return self.div(other)
    def __rtruediv__(self, other: "Scalar | STSequence"): return other * self.reciprocal()
    def __itruediv__(self, other: "Scalar | STSequence") -> T.Self:
        self.div_(other)
        return self

    def __floordiv__(self, other: STOrSTSequence): return self.floor_divide(other)
    #def __rfloordiv__(self, other: "TensorList"): return other.floor_divide(self)
    def __ifloordiv__(self, other: STOrSTSequence):
        self.floor_divide_(other)
        return self

    def __mod__(self, other: STOrSTSequence): return self.remainder(other)
    #def __rmod__(self, other: STOrSTSequence): return self.remainder(other)
    def __imod__(self, other: STOrSTSequence):
        self.remainder_(other)
        return self

    def __pow__(self, other: "Scalar | STSequence"): return self.pow(other)
    #def __rpow__(self, other: Scalar | STSequence): return self.pow(other)
    def __ipow__(self, other: "Scalar | STSequence"):
        self.pow_(other)
        return self

    def __neg__(self): return self.neg()

    def __eq__(self, other: STOrSTSequence): return self.eq(other) # type:ignore
    def __ne__(self, other: STOrSTSequence): return self.ne(other) # type:ignore
    def __lt__(self, other: STOrSTSequence): return self.lt(other) # type:ignore
    def __le__(self, other: STOrSTSequence): return self.le(other) # type:ignore
    def __gt__(self, other: STOrSTSequence): return self.gt(other) # type:ignore
    def __ge__(self, other: STOrSTSequence): return self.ge(other) # type:ignore

    def __invert__(self): return self.map(torch.logical_not)


    def map(self, fn: A.Callable[..., torch.Tensor], *args, **kwargs):
        """Applies `fn` to all elements of this TensorList
        and returns a new TensorList with return values of the callable."""
        return self.__class__(fn(i, *args, **kwargs) for i in self)
    def map_inplace_(self, fn: A.Callable[..., T.Any], *args, **kwargs):
        """Applies an in-place `fn` to all elements of this TensorList."""
        for i in self: fn(i, *args, **kwargs)
        return self

    def filter(self, fn: A.Callable[..., bool], *args, **kwargs):
        """Returns a TensorList with all elements for which `fn` returned True."""
        return self.__class__(i for i in self if fn(i, *args, **kwargs))

    def zipmap(self, fn: A.Callable, other: T.Any | list | tuple, *args, **kwargs):
        """If `other` is list/tuple, applies `fn` to this TensorList zipped with `other`.
        Otherwise applies `fn` to this TensorList and `other`.
        Returns a new TensorList with return values of the callable."""
        if isinstance(other, (list, tuple)): return self.__class__(fn(i, j, *args, **kwargs) for i, j in zip(self, other))
        return self.__class__(fn(i, other, *args, **kwargs) for i in self)

    def zipmap_inplace_(self, fn: A.Callable[..., T.Any], other: T.Any | list | tuple, *args, **kwargs):
        """If `other` is list/tuple, applies `fn` to this TensorList zipped with `other`.
        Otherwise applies `fn` to this TensorList and `other`.
        The callable must modify elements in-place."""
        if isinstance(other, (list, tuple)):
            for i, j in zip(self, other): fn(i, j, *args, **kwargs)
        else:
            for i in self: fn(i, other, *args, **kwargs)
        return self

    def zipmap_args(self, fn: A.Callable[..., T.Any], *others, **kwargs):
        """If `args` is list/tuple, applies `fn` to this TensorList zipped with `other`.
        Otherwise applies `fn` to this TensorList and `other`."""
        if isinstance(others[0], (list, tuple)):
            return self.__class__(fn(*z, **kwargs) for z in zip(self, *others))
        else:
            return self.__class__(fn(i, *others, **kwargs) for i in self)

    def zipmap_args_inplace_(self, fn: A.Callable[..., T.Any], *others, **kwargs):
        """If `args` is list/tuple, applies `fn` to this TensorList zipped with `other`.
        Otherwise applies `fn` to this TensorList and `other`.
        The callable must modify elements in-place."""
        if isinstance(others[0], (list, tuple)):
            for z in zip(self, *others): fn(*z, **kwargs)
        else:
            for i in self: fn(i, *others, **kwargs)
        return self

    def _foreach_apply(self, fn: A.Callable[[list[torch.Tensor]], list[torch.Tensor]], *args, **kwargs):
        """Applies a torch._foreach_xxx function to self and converts returned list back to TensorList or subclass."""
        return self.__class__(fn(self), *args, **kwargs)

    # def __getattr__(self, name: str) -> A.Callable:
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

    # apparently I can't use float for typing if I call a method "float"
    def as_float(self): return self.__class__(i.float() for i in self)
    def as_bool(self): return self.__class__(i.bool() for i in self)
    def as_int(self): return self.__class__(i.int() for i in self)

    def copy_(self, src: TensorSequence, non_blocking = False):
        """Copies the elements from src tensors into self tensors."""
        torch._foreach_copy_(self, src, non_blocking=non_blocking)
    def set_(self, storage: A.Iterable[torch.Tensor | torch.types.Storage]):
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

    def empty_like(self, **kwargs: T.Unpack[_NewTensorKwargs]): return self.__class__(torch.empty_like(i, **kwargs) for i in self)
    def zeros_like(self, **kwargs: T.Unpack[_NewTensorKwargs]): return self.__class__(torch.zeros_like(i, **kwargs) for i in self)
    def ones_like(self, **kwargs: T.Unpack[_NewTensorKwargs]): return self.__class__(torch.ones_like(i, **kwargs) for i in self)
    def full_like(self, fill_value: "Scalar | ScalarSequence", **kwargs: T.Unpack[_NewTensorKwargs]):
        #return self.__class__(torch.full_like(i, fill_value=fill_value, **kwargs) for i in self)
        return self.zipmap_args(torch.full_like, fill_value, **kwargs)

    def rand_like(self, **kwargs: T.Unpack[_NewTensorKwargs]): return self.__class__(torch.rand_like(i, **kwargs) for i in self)
    def randn_like(self, **kwargs: T.Unpack[_NewTensorKwargs]): return self.__class__(torch.randn_like(i, **kwargs) for i in self)

    def randint_like(self, low: "Scalar | ScalarSequence", high: "Scalar | ScalarSequence", **kwargs: T.Unpack[_NewTensorKwargs]):
        return self.zipmap_args(torch.randint_like, low, high, **kwargs)
    def uniform_like(self, low: "Scalar | ScalarSequence" = 0, high: "Scalar | ScalarSequence" = 1, **kwargs: T.Unpack[_NewTensorKwargs]):
        res = self.empty_like(**kwargs)
        res.uniform_(low, high)
        return res
    def sphere_like(self, radius: "Scalar | ScalarSequence", **kwargs: T.Unpack[_NewTensorKwargs]) -> T.Self:
        r = self.randn_like(**kwargs)
        return (r * radius) / r.total_vector_norm() # type:ignore
    def bernoulli(self):
        return self.__class__(torch.bernoulli(i) for i in self)
    def bernoulli_like(self, p: "Scalar | ScalarSequence" = 0.5):
        """p is probability of a 1, other values will be 0."""
        return self.__class__(torch.bernoulli(i) for i in self.full_like(p))
    def rademacher_like(self, p: "Scalar | ScalarSequence" = 0.5):
        """p is probability of a 1, other values will be -1."""
        return self.bernoulli_like(p) * 2 - 1

    def sample_like(self, eps: "Scalar | ScalarSequence" = 1, distribution: Distributions = 'normal'):
        """Sample around 0."""
        if distribution == 'normal': return self.randn_like() * eps
        elif distribution == 'uniform':
            if isinstance(eps, (list,tuple)):
                return self.uniform_like([-i/2 for i in eps], [i/2 for i in eps]) # type:ignore
            else:
                return self.uniform_like(-eps/2, eps/2)
        elif distribution == 'sphere': return self.sphere_like(eps)
        elif distribution == 'rademacher': return self.rademacher_like() * eps

    def eq(self, other: STOrSTSequence): return self.zipmap(torch.eq, other)
    def eq_(self, other: STOrSTSequence): return self.zipmap_inplace_(MethodCallerWithArgs('eq_'), other)
    def ne(self, other: STOrSTSequence): return self.zipmap(torch.ne, other)
    def ne_(self, other: STOrSTSequence): return self.zipmap_inplace_(MethodCallerWithArgs('ne_'), other)
    def lt(self, other: STOrSTSequence): return self.zipmap(torch.lt, other)
    def lt_(self, other: STOrSTSequence): return self.zipmap_inplace_(MethodCallerWithArgs('lt_'), other)
    def le(self, other: STOrSTSequence): return self.zipmap(torch.le, other)
    def le_(self, other: STOrSTSequence): return self.zipmap_inplace_(MethodCallerWithArgs('le_'), other)
    def gt(self, other: STOrSTSequence): return self.zipmap(torch.gt, other)
    def gt_(self, other: STOrSTSequence): return self.zipmap_inplace_(MethodCallerWithArgs('gt_'), other)
    def ge(self, other: STOrSTSequence): return self.zipmap(torch.ge, other)
    def ge_(self, other: STOrSTSequence): return self.zipmap_inplace_(MethodCallerWithArgs('ge_'), other)

    def equal(self, other: STOrSTSequence): return self.zipmap(torch.equal, other)

    def add(self, other: STOrSTSequence, alpha: Scalar = 1):
        if alpha == 1: return self.__class__(torch._foreach_add(self, other))
        else: return self.__class__(torch._foreach_add(self, other, alpha = alpha)) # type:ignore
    def add_(self, other: STOrSTSequence, alpha: Scalar = 1):
        if alpha == 1: torch._foreach_add_(self, other)
        else: torch._foreach_add_(self, other, alpha = alpha) # type:ignore
        return self


    def sub(self, other: "Scalar | STSequence", alpha: Scalar = 1):
        if alpha == 1: return self.__class__(torch._foreach_sub(self, other))
        else: return self.__class__(torch._foreach_sub(self, other, alpha = alpha)) # type:ignore
    def sub_(self, other: "Scalar | STSequence", alpha: Scalar = 1):
        if alpha == 1: torch._foreach_sub_(self, other)
        else: torch._foreach_sub_(self, other, alpha = alpha) # type:ignore
        return self

    def neg(self): return self.__class__(torch._foreach_neg(self))
    def neg_(self):
        torch._foreach_neg_(self)
        return self

    def mul(self, other: STOrSTSequence): return self.__class__(torch._foreach_mul(self, other))
    def mul_(self, other: STOrSTSequence):
        torch._foreach_mul_(self, other)
        return self

    def div(self, other: STOrSTSequence) -> T.Self: return self.__class__(torch._foreach_div(self, other))
    def div_(self, other: STOrSTSequence):
        torch._foreach_div_(self, other)
        return self

    def pow(self, exponent: "Scalar | STSequence"): return self.__class__(torch._foreach_pow(self, exponent))
    def pow_(self, exponent: "Scalar | STSequence"):
        torch._foreach_pow_(self, exponent)
        return self

    def sqrt(self): return self.__class__(torch._foreach_sqrt(self))
    def sqrt_(self):
        torch._foreach_sqrt_(self)
        return self

    def remainder(self, other: STOrSTSequence): return self.zipmap(torch.remainder, other)
    def remainder_(self, other: STOrSTSequence): return self.zipmap_inplace_(MethodCallerWithArgs('remainder_'), other)

    def floor_divide(self, other: STOrSTSequence): return self.zipmap(torch.floor_divide, other)
    def floor_divide_(self, other: STOrSTSequence): return self.zipmap_inplace_(MethodCallerWithArgs('floor_divide_'), other)

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

    def max(self, dim = None, keepdim = False):
        if dim is None and not keepdim: return self.__class__(torch._foreach_max(self))
        return self.__class__(i.max(dim=dim, keepdim=keepdim) for i in self)

    def min(self, dim = None, keepdim = False):
        if dim is None and not keepdim: return self.__class__(torch._foreach_max(self.neg())).neg()
        return self.__class__(i.min(dim=dim, keepdim=keepdim) for i in self)

    def mean(self, dim = None, keepdim = False): return self.__class__(i.mean(dim=dim, keepdim=keepdim) for i in self)
    def sum(self, dim = None, keepdim = False): return self.__class__(i.sum(dim=dim, keepdim=keepdim) for i in self)
    def prod(self, dim = None, keepdim = False): return self.__class__(i.prod(dim=dim, keepdim=keepdim) for i in self)

    def clamp_min(self, other: "Scalar | STSequence"): return self.__class__(torch._foreach_clamp_min(self, other))
    def clamp_min_(self, other: "Scalar | STSequence"):
        torch._foreach_clamp_min_(self, other)
        return self
    def clamp_max(self, other: "Scalar | STSequence"): return self.__class__(torch._foreach_clamp_max(self, other))
    def clamp_max_(self, other: "Scalar | STSequence"):
        torch._foreach_clamp_max_(self, other)
        return self

    def clamp(self, min: "Scalar | STSequence | None" = None, max: "Scalar | STSequence | None" = None):
        l = self
        if min is not None: l = l.clamp_min(min)
        if max is not None: l = l.clamp_max(max)
        return l
    def clamp_(self, min: "Scalar | STSequence | None" = None, max: "Scalar | STSequence | None" = None):
        if min is not None: self.clamp_min_(min)
        if max is not None: self.clamp_max_(max)
        return self

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

    def lerp(self, tensors1: TensorSequence, weight: "Scalar | TensorSequence"):
        return self.__class__(torch._foreach_lerp(self, tensors1, weight))
    def lerp_(self, tensors1: TensorSequence, weight: "Scalar | TensorSequence"):
        torch._foreach_lerp_(self, tensors1, weight)
        return self


    def addcmul(self, tensors1: TensorSequence, tensor2: TensorSequence, value: "Scalar | A.Sequence[Scalar] | torch.Tensor"):
        return self.__class__(torch._foreach_addcmul(self, tensors1, tensor2, value))
    def addcmul_(self, tensors1: TensorSequence, tensor2: TensorSequence, value: "Scalar | A.Sequence[Scalar] | torch.Tensor"):
        torch._foreach_addcmul_(self, tensors1, tensor2, value)
        return self
    def addcdiv(self, tensors1: TensorSequence, tensor2: TensorSequence, value: "Scalar | A.Sequence[Scalar] | torch.Tensor"):
        return self.__class__(torch._foreach_addcdiv(self, tensors1, tensor2, value))
    def addcdiv_(self, tensors1: TensorSequence, tensor2: TensorSequence, value: "Scalar | A.Sequence[Scalar] | torch.Tensor"):
        torch._foreach_addcdiv_(self, tensors1, tensor2, value)
        return self

    def uniform_(self, low: "Scalar | ScalarSequence" = 0, high: "Scalar | ScalarSequence" = 1, generator = None):
        return self.zipmap_args_inplace_(MethodCallerWithArgs('uniform_'), low, high, generator = generator)

    def squeeze(self, dim = None):
        return self.__class__(i.squeeze(dim) for i in self)

    def squeeze_(self, dim = None):
        for i in self: i.squeeze_(dim)
        return self

    def nan_to_num_(self,nan: float | None = None,posinf: float | None = None,neginf: float | None = None):
        for i in self: torch.nan_to_num_(i, nan = nan, posinf = posinf, neginf = neginf)
        return self

    def ravel(self): return self.__class__(i.ravel() for i in self)

    def any(self): return self.__class__(i.any() for i in self)
    def all(self): return self.__class__(i.all() for i in self)
    def isfinite(self): return self.__class__(i.isfinite() for i in self)

    def flatiter(self) -> A.Generator[torch.Tensor]:
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
        for i in range(len(self)): self[i] = res[i]
        return self

    def add(self, other: STOrSTSequence, alpha: Scalar = 1):
        if alpha == 1: return self.zipmap_args(operator.add, other)
        else: return self.zipmap_args(_alpha_add, other, alpha = alpha)
    def add_(self, other: STOrSTSequence, alpha: Scalar = 1):
        return self._set_to_method_result('add', other, alpha = alpha)

    def sub(self, other: "Scalar | STSequence", alpha: Scalar = 1):
        if alpha == 1: return self.zipmap_args(operator.sub, other)
        else: return self.zipmap_args(_alpha_add, other, alpha = -alpha)
    def sub_(self, other: "Scalar | STSequence", alpha: Scalar = 1):
        return self._set_to_method_result('sub', other, alpha = alpha)

    def neg(self): return self.__class__(-i for i in self)
    def neg_(self): return self._set_to_method_result('neg')

    def mul(self, other: STOrSTSequence): return self.zipmap_args(operator.mul, other)
    def mul_(self, other: STOrSTSequence): return self._set_to_method_result('mul', other)

    def div(self, other: STOrSTSequence) -> T.Self: return self.zipmap_args(operator.truediv, other)
    def div_(self, other: STOrSTSequence): return self._set_to_method_result('div', other)

    def pow(self, exponent: "Scalar | STSequence"): return self.zipmap_args(math.pow, exponent)
    def pow_(self, exponent: "Scalar | STSequence"): return self._set_to_method_result('pow_', exponent)

    def __rtruediv__(self, other: "Scalar | STSequence"):
        # overriding because TensorList implements this through reciprocal
        if isinstance(other, (tuple,list)): return self.__class__(o / i for o, i in zip(self, other))
        return self.__class__(other / i for i in self)

def stack(tensorlists: A.Iterable[TensorList], dim = 0):
    """Returns a tensorlist with the same elements as the input tensorlists, but stacked along the specified dimension."""
    return TensorList(torch.stack(i, dim = dim) for i in zip(*tensorlists))

def mean(tensorlists: A.Iterable[TensorList]):
    """Returns a tensorlist which is the mean of given tensorlists."""
    return stack(tensorlists).mean(0)

def sum(tensorlists: A.Iterable[TensorList]):
    """Returns a tensorlist which is the sum of given tensorlists."""
    return stack(tensorlists).sum(0)
