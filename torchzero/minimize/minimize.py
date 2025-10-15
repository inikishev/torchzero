"""WIP API"""
import itertools
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, NamedTuple, cast, overload

import numpy as np
import torch

from ..core import Module, Optimizer
from ..utils import tofloat
from .methods import _get_method_from_str

_fn_autograd = Callable[[torch.Tensor], torch.Tensor | Any]
_fn_custom_grad = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
_scalar = float | np.ndarray | torch.Tensor
_method = str | Module | Sequence[Module] | Callable[..., torch.optim.Optimizer]

def _tensorlist_norm(tensors: Iterable[torch.Tensor], ord) -> torch.Tensor:
    """returns a scalar - global norm of tensors"""
    if ord == torch.inf:
        return max(torch._foreach_max(torch._foreach_abs(tuple(tensors))))

    if ord == 1:
        return cast(torch.Tensor, sum(t.abs().sum() for t in tensors))

    if ord % 2 != 0:
        tensors = torch._foreach_abs(tuple(tensors))

    tensors = torch._foreach_pow(tuple(tensors), ord)
    return sum(t.sum() for t in tensors) ** (1 / ord)



class Params:
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
        return Params(
            args = [a.clone() for a in self.args],
            kwargs={k:v.clone() for k,v in self.kwargs.items()}
        )

    def __repr__(self):
        if len(self.args) == 1 and len(self.kwargs) == 0:
            return f"Params({repr(self.x)})"

        s = "Params("
        if len(self.args) > 0:
            s = f"{s}\n\targs = (\n\t\t"
            s += ",\n\t\t".join(str(a) for a in self.args)
            s = s + "\n\t)"

        if len(self.kwargs) > 0:
            s = f'{s}\n\tkwargs = (\n\t\t'
            for k,v in self.kwargs.items():
                s = f"{s}{k}={v},\n\t\t"
            s = s[:-2] + "\t)"

        return f"{s}\n)"

    def _call(self, f):
        return f(*self.args, **self.kwargs)

    def _detach_clone(self):
        return Params(
            args = [a.detach().clone() for a in self.args],
            kwargs={k:v.detach().clone() for k,v in self.kwargs.items()}
        )

    def _detach_cpu_clone(self):
        return Params(
            args = [a.detach().cpu().clone() for a in self.args],
            kwargs={k:v.detach().cpu().clone() for k,v in self.kwargs.items()}
        )

    def _requires_grad_(self, mode=True):
        return Params(
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
    Params
)



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
    def __init__(self, f: _fn_autograd | _fn_custom_grad, x0: Params, reduce_fn: Callable, max_history,
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

    def __call__(self, x: Params, g: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
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
    params: Params
    x: torch.Tensor | None
    success: bool
    message: str
    fun: float
    n_iters: int
    n_evals: int
    g_norm: torch.Tensor | None
    dir_norm: torch.Tensor | None
    losses: list[float]
    history: deque[tuple[torch.Tensor, torch.Tensor]]

    def __repr__(self):
        newline = "\n"
        ident = " " * 10
        return (
            f"message:  {self.message}\n"
            f"success:  {self.success}\n"
            f"fun:      {self.fun}\n"
            f"params:   {repr(self.params).replace(newline, newline+ident)}\n"
            f"x:        {self.x}\n"
            f"n_iters:  {self.n_iters}\n"
            f"n_evals:  {self.n_evals}\n"
            f"g_norm:   {self.g_norm}\n"
            f"dir_norm: {self.dir_norm}\n"
        )



def _make_params(x0: _x0):
    x = cast(Any, x0)

    # kwargs
    if isinstance(x, Params): return x

    # single tensor
    if isinstance(x, torch.Tensor): return Params(args = (x, ), kwargs = {})

    if isinstance(x, Sequence):
        # args
        if isinstance(x[0], torch.Tensor): return Params(args=x, kwargs = {})

        # tuple of (args, kwrgs)
        assert len(x) == 2 and isinstance(x[0], Sequence) and isinstance(x[1], Mapping)
        return Params(args=x[0], kwargs=x[1])

    if isinstance(x, Mapping):
        # dict with args and kwargs
        if "args" in x or "kwargs" in x: return Params(args=x.get("args", ()), kwargs=x.get("kwargs", {}))

        # kwargs
        return Params(args=(), kwargs=x)

    raise TypeError(type(x))


def minimize(
    f: _fn_autograd | _fn_custom_grad,
    x0: _x0,

    method: _method | None = None,

    maxeval: int | None = None,
    maxiter: int | None = None,
    maxsec: float | None = None,
    ftol: _scalar | None = None,
    gtol: _scalar | None = 1e-5,
    xtol: _scalar | None = None,
    max_no_improvement_iters: int | None = 100,

    reduce_fn: Callable[[torch.Tensor], torch.Tensor] = torch.sum,
    max_history: int = 0,

    custom_grad: bool = False,
    use_termination_exceptions: bool = True,
    norm = torch.inf,

) -> MinimizeResult:
    """Minimize a scalar or multiobjective function of one or more variables.

    Args:
        f (_fn_autograd | _fn_custom_grad):
            The objective function to be minimized.
        x0 (_x0):
            Initial guess. Can be torch.Tensor, tuple of torch.Tensors to pass as args,
            or dictionary of torch.Tensors to pass as kwargs.
        method (_method | None, optional):
            Type of solver. Can be a string, a ``Module`` (like ``tz.m.BFGS()``), or a list of ``Module``.
            By default chooses BFGS or L-BFGS depending on number of variables. Defaults to None.
        maxeval (int | None, optional):
            terminate when exceeded this number of function evaluations. Defaults to None.
        maxiter (int | None, optional):
            terminate when exceeded this number of solver iterations,
            each iteration may perform multiple function evaluations. Defaults to None.
        maxsec (float | None, optional):
            terminate after optimizing for this many seconds. Defaults to None.
        ftol (_scalar | None, optional):
            terminate when reached a solution with objective value less or equal to this value. Defaults to None.
        gtol (_scalar | None, optional):
            terminate when gradient norm is less or equal to this value.
            The type of norm is controlled by ``norm`` argument and is infinity norm by default. Defaults to 1e-5.
        xtol (_scalar | None, optional):
            terminate when norm of difference between successive parameters is less or equal to this value. Defaults to None.
        max_no_improvement_iters (int | None, optional):
            terminate when objective value hasn't improved once for this many consecutive iterations. Defaults to 100.
        reduce_fn (Callable[[torch.Tensor], torch.Tensor], optional):
            only has effect when ``f`` is multi-objective / least-squares. Determines how to convert
            vector returned by ``f`` to a single scalar value for ``ftol`` and ``max_no_improvement_iters``.
            Defaults to torch.sum.
        max_history (int, optional):
            stores this many last evaluated parameters and their values.
            Set to -1 to store all parameters. Set to 0 to store nothing (default).
        custom_grad (bool, optional):
            Allows specifying a custom gradient function instead of using autograd.
            if True, objective function ``f`` must of the following form:
            ```python
            def f(x, grad):
                value = objective(x)
                if grad.numel() > 0:
                    grad[:] = objective_gradient(x)
                return value
            ```

            Defaults to False.
        use_termination_exceptions (bool, optional):
            if True, ``maxeval`` and ``maxsec`` use exceptions to terminate, therefore they are able to trigger
            mid-iteration. If False, they can only trigger after iteration, so it might perform slightly more
            evals and for slightly more seconds than requested. Defaults to True.
        norm (float, optional):
            type of norm to use for gradient and update tolerances. Defaults to torch.inf.

    Raises:
        RuntimeError: _description_

    Returns:
        MinimizeResult: _description_
    """

    x0 = _make_params(x0)
    x = x0._requires_grad_(True)

    # checks
    if custom_grad:
        if not (len(x.args) == 1 and len(x.kwargs) == 0):
            raise RuntimeError("custom_grad only works when `x` is a single tensor.")

    # determine method if None
    if method is None:
        max_dim = 5_000 if next(iter(x.parameters())).is_cuda else 1_000
        if sum(p.numel() for p in x.parameters()) > max_dim: method = 'lbfgs'
        else: method = 'bfgs'

    opt_fn = _get_opt_fn(method)
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

    tiny = torch.finfo(list(x0.parameters())[0].dtype).tiny ** 2
    if gtol == 0: gtol = tiny
    if xtol == 0: xtol = tiny

    p_prev = None if xtol is None else [p.detach().clone() for p in x.parameters()]
    fmin = float("inf")
    niter = 0
    n_no_improvement = 0
    g_norm = None
    dir_norm = None

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
                if grads is not None:
                    g_norm = _tensorlist_norm(grads, norm)
                    if g_norm <= gtol:
                        terminate_msg = 'gradient norm is below tolerance'
                        success = True
                        break

                # due to the way torchzero works we sometimes don't populate .grad,
                # e.g. with Newton, therefore fallback on xtol
                else:
                    if xtol is None: xtol = tiny

            # difference in parameters
            if xtol is not None:
                p_new = [p.detach().clone() for p in x.parameters()]

                if p_prev is None: # happens when xtol is set in gtol logic
                    p_prev = p_new

                else:
                    dir_norm = _tensorlist_norm(torch._foreach_sub(p_new, p_prev), norm)
                    if dir_norm <= xtol:
                        terminate_msg = 'update norm is below tolerance'
                        success = True
                        break

                    p_prev = p_new

            # no improvement steps
            if max_no_improvement_iters is not None:
                if f_wrapped.fmin >= fmin:
                    n_no_improvement += 1
                else:
                    fmin = f_wrapped.fmin
                    n_no_improvement = 0

                if n_no_improvement >= max_no_improvement_iters:
                    terminate_msg = 'reached maximum steps without improvement'
                    success = False
                    break

    history=f_wrapped.history
    if history is None: history = deque()

    x_vec = None
    if len(x0.args) == 1 and len(x0.kwargs) == 0:
        x_vec = f_wrapped.x_best.x

    result = MinimizeResult(
        params = f_wrapped.x_best,
        x = x_vec,
        success = success,
        message = terminate_msg,
        fun = f_wrapped.fmin,
        n_iters = niter,
        n_evals = f_wrapped.evals,
        g_norm = g_norm,
        dir_norm = dir_norm,
        losses = losses,
        history = history,
    )

    return result


