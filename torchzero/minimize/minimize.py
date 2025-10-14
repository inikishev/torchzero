"""WIP API"""
import itertools
import time
from collections import deque
from collections.abc import Callable, Sequence, Mapping, Iterable
from typing import Any, NamedTuple, cast, overload

import numpy as np
import torch

from .. import m
from ..core import Module, Optimizer
from ..utils import tofloat

_fn_autograd = Callable[[torch.Tensor], torch.Tensor | Any]
_fn_custom_grad = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
_scalar = float | np.ndarray | torch.Tensor
_method = str | Module | Sequence[Module] | Callable[..., torch.optim.Optimizer]

def _tensorlist_norm(tensors: Iterable[torch.Tensor], ord):
    """returns a scalar - global norm of tensors"""
    if ord == torch.inf:
        return max(torch._foreach_max(torch._foreach_abs(tuple(tensors))))

    if ord == 1:
        return sum(t.abs().sum() for t in tensors)

    if ord % 2 != 0:
        tensors = torch._foreach_abs(tuple(tensors))

    tensors = torch._foreach_pow(tuple(tensors), ord)
    return sum(t.sum() for t in tensors) ** (1 / ord)



class Kwargs:
    __slots__ = ("args", "kwargs")
    def __init__(self, args: Sequence[torch.Tensor], kwargs: Mapping[str, torch.Tensor]):
        self.args = tuple(args)
        self.kwargs = dict(kwargs)

    @property
    def x(self):
        assert len(self.args) == 1
        assert len(self.kwargs) == 0
        return self.args[0]

    def parameters(self):
        yield from self.args
        yield from self.kwargs.values()

    def clone(self):
        return Kwargs(
            args = [a.clone() for a in self.args],
            kwargs={k:v.clone() for k,v in self.kwargs.items()}
        )

    def _call(self, f):
        return f(*self.args, **self.kwargs)

    def _detach_clone(self):
        return Kwargs(
            args = [a.detach().clone() for a in self.args],
            kwargs={k:v.detach().clone() for k,v in self.kwargs.items()}
        )

    def _detach_cpu_clone(self):
        return Kwargs(
            args = [a.detach().cpu().clone() for a in self.args],
            kwargs={k:v.detach().cpu().clone() for k,v in self.kwargs.items()}
        )

    def _requires_grad_(self, mode=True):
        return Kwargs(
            args = [a.requires_grad_(mode) for a in self.args],
            kwargs={k:v.requires_grad_(mode) for k,v in self.kwargs.items()}
        )


    def _grads(self):
        params = tuple(self.parameters())
        if all(p.grad is None for p in params): return None
        return [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]


_x0 = (
    torch.Tensor |
    Sequence[torch.Tensor] |
    Mapping[str, torch.Tensor] |
    Mapping[str, Sequence[torch.Tensor] | Mapping[str, torch.Tensor]] |
    tuple[Sequence[torch.Tensor], Mapping[str, torch.Tensor]] |
    Sequence[Sequence[torch.Tensor] | Mapping[str, torch.Tensor]] |
    Kwargs
)

def _get_method_from_str(method: str) -> list[Module]:
    method = ''.join(c for c in method.lower().strip() if c.isalnum())
    if method == "bfgs":
        return [m.BFGS(), m.Backtracking()]

    raise NotImplementedError(method)

def _get_opt_fn(method: _method):
    if isinstance(method, str):
        return lambda p: Optimizer(p, *_get_method_from_str(method))

    if isinstance(method, Module):
        return lambda p: Optimizer(p, method)

    if isinstance(method, Sequence):
        return lambda p: Optimizer(p, *method)

    if callable(method):
        return method

    raise ValueError(method)

def _is_scalar(x):
    if isinstance(x, torch.Tensor): return x.numel() == 1
    if isinstance(x, np.ndarray): return x.size == 1
    return True

def _maybe_detach_cpu(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu()
    return x

class _MaxEvaluationsReached(Exception): pass
class _MaxSecondsReached(Exception): pass
class Terminate(Exception): pass

class _WrappedFunc:
    def __init__(self, f: _fn_autograd | _fn_custom_grad, x0: Kwargs, reduce_fn: Callable, max_history,
                 maxeval:int | None, maxsec: float | None, custom_grad:bool):
        self.f = f
        self.maxeval = maxeval
        self.reduce_fn = reduce_fn
        self.custom_grad = custom_grad
        self.maxsec = maxsec

        self.x_best = x0.clone()
        self.fmin = float("inf")
        self.evals = 0
        self.start = time.time()

        if max_history == -1: max_history = None # unlimited history
        if max_history == 0: self.history = None
        else: self.history = deque(maxlen=max_history)

    def __call__(self, x: Kwargs, g: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.maxeval is not None and self.evals >= self.maxeval:
            raise _MaxEvaluationsReached

        if self.maxsec is not None and time.time() - self.start >= self.maxsec:
            raise _MaxSecondsReached

        self.evals += 1

        if self.custom_grad:
            assert g is not None
            assert len(x.args) == 1 and len(x.kwargs) == 0
            v = v_scalar = cast(_fn_custom_grad, self.f)(x.x, g)
        else:
            v = v_scalar = x._call(self.f)

        with torch.no_grad():

            # multi-value v, reduce using reduce func
            if isinstance(v, torch.Tensor) and v.numel() > 1:
                v_scalar = self.reduce_fn(v)

            if v_scalar < self.fmin:
                self.fmin = tofloat(v_scalar)
                self.x_best = x._detach_clone()

            if self.history is not None:
                self.history.append((x._detach_cpu_clone(), _maybe_detach_cpu(v)))

        return v, g

class MinimizeResult(NamedTuple):
    kwargs: Kwargs
    x: torch.Tensor | None
    success: bool
    message: str
    fun: float
    n_iters: int
    n_evals: int
    losses: list[float]
    history: deque[tuple[torch.Tensor, torch.Tensor]]



def _make_kwargs(x0: _x0):
    x = cast(Any, x0)

    # kwargs
    if isinstance(x, Kwargs): return x

    # single tensor
    if isinstance(x, torch.Tensor): return Kwargs(args = (x, ), kwargs = {})

    if isinstance(x, Sequence):
        # args
        if isinstance(x[0], torch.Tensor): return Kwargs(args=x, kwargs = {})

        # tuple of (args, kwrgs)
        assert len(x) == 2 and isinstance(x[0], Sequence) and isinstance(x[1], Mapping)
        return Kwargs(args=x[0], kwargs=x[1])

    if isinstance(x, Mapping):
        # dict with args and kwargs
        if "args" in x or "kwargs" in x: return Kwargs(args=x.get("args", ()), kwargs=x.get("kwargs", {}))

        # kwargs
        return Kwargs(args=(), kwargs=x)

    raise TypeError(type(x))


def minimize(
    f: _fn_autograd | _fn_custom_grad,
    x0: _x0,

    method: _method,

    maxeval: int | None = None,
    maxiter: int | None = None,
    maxsec: float | None = None,
    ftol: _scalar | None = None,
    gtol: _scalar | None = 0,
    xtol: _scalar | None = None,
    max_no_improvement_steps: int | None = 100,

    reduce_fn: Callable[[torch.Tensor], torch.Tensor] = torch.sum,
    max_history: int = 0,

    custom_grad: bool = False,
    use_termination_exceptions: bool = True,
    norm = torch.inf

) -> MinimizeResult:
    opt_fn = _get_opt_fn(method)
    x0 = _make_kwargs(x0)
    x = x0._requires_grad_(True)

    optimizer = opt_fn(list(x.parameters()))

    f_wrapped = _WrappedFunc(
        f,
        x0=x0,
        reduce_fn=reduce_fn,
        max_history=max_history,
        maxeval=maxeval,
        custom_grad=custom_grad,
        maxsec=maxsec,
    )

    def closure(backward=True):

        g = None
        v = None
        if custom_grad:
            v = x.x
            if backward: g = torch.empty_like(v)
            else: g = torch.empty(0, device=v.device, dtype=v.dtype)

        loss, g = f_wrapped(x, g=g)

        if backward:

            # custom gradients provided by user
            if g is not None:
                assert v is not None
                v.grad = g

            # autograd
            else:
                optimizer.zero_grad()
                loss.backward()

        return loss

    losses = []

    p_prev = None if xtol is None else [p.detach().clone() for p in x.parameters()]
    fmin = float("inf")
    niter = 0
    n_no_improvement = 0

    terminate_msg = "max iterations reached"
    success = False

    exceptions: list | tuple = [Terminate]
    if use_termination_exceptions:
        if maxeval is not None: exceptions.append(_MaxEvaluationsReached)
        if maxsec is not None: exceptions.append(_MaxSecondsReached)
    exceptions = tuple(exceptions)

    for i in (range(maxiter) if maxiter is not None else itertools.count()):
        niter += 1

        # ----------------------------------- step ----------------------------------- #
        try:
            v = v_scalar = optimizer.step(closure) # pyright:ignore[reportCallIssue,reportArgumentType]
        except exceptions:
            break

        with torch.no_grad():
            assert v is not None and v_scalar is not None

            if isinstance(v, torch.Tensor) and v.numel() > 1:
                v_scalar = reduce_fn(v)

            losses.append(tofloat(v_scalar))

            # --------------------------- termination criteria --------------------------- #

            # termination criteria on optimizer
            if isinstance(optimizer, Optimizer) and optimizer.should_terminate:
                terminate_msg = 'optimizer-specific termination criteria triggered'
                success = True
                break

            # max seconds (when use_termination_exceptions=False)
            if maxsec is not None and time.time() - f_wrapped.start >= maxsec:
                terminate_msg = 'max seconds reached'
                success = False
                break

            # max evals (when use_termination_exceptions=False)
            if maxeval is not None and f_wrapped.evals >= maxeval:
                terminate_msg = 'max evaluations reached'
                success = False
                break

            # min function value
            if ftol is not None and v_scalar <= ftol:
                terminate_msg = 'target function value reached'
                success = True
                break

            # gradient infinity norm
            if gtol is not None:
                grads = x._grads()
                if grads is not None and _tensorlist_norm(grads, norm) <= gtol:
                    terminate_msg = 'gradient norm is below tolerance'
                    success = True
                    break

            # difference in parameters
            if xtol is not None:
                assert p_prev is not None
                p_new = [p.detach().clone() for p in x.parameters()]

                if _tensorlist_norm(torch._foreach_sub(p_new, p_prev), norm) <= xtol:
                    terminate_msg = 'update norm is below tolerance'
                    success = True
                    break

                p_prev = p_new

            # no improvement steps
            if max_no_improvement_steps is not None:

                if fmin >= f_wrapped.fmin: n_no_improvement += 1
                else: n_no_improvement = 0

                if n_no_improvement >= max_no_improvement_steps:
                    terminate_msg = 'reached maximum steps without improvement'
                    success = False
                    break

    history=f_wrapped.history
    if history is None: history = deque()

    x_vec = None
    if len(x0.args) == 1 and len(x0.kwargs) == 0:
        x_vec = f_wrapped.x_best.x

    result = MinimizeResult(
        kwargs = f_wrapped.x_best,
        x = x_vec,
        success = success,
        message = terminate_msg,
        fun = f_wrapped.fmin,
        n_iters = niter,
        n_evals = f_wrapped.evals,
        losses = losses,
        history = history,
    )

    return result


