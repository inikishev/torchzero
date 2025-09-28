import warnings
from typing import Literal, Any
from collections.abc import Mapping, Callable
from functools import partial
import numpy as np
import torch

import nlopt
from ...utils import TensorList
from .wrapper import WrapperBase

_ALGOS_LITERAL = Literal[
    "GN_DIRECT",  # = _nlopt.GN_DIRECT
    "GN_DIRECT_L",  # = _nlopt.GN_DIRECT_L
    "GN_DIRECT_L_RAND",  # = _nlopt.GN_DIRECT_L_RAND
    "GN_DIRECT_NOSCAL",  # = _nlopt.GN_DIRECT_NOSCAL
    "GN_DIRECT_L_NOSCAL",  # = _nlopt.GN_DIRECT_L_NOSCAL
    "GN_DIRECT_L_RAND_NOSCAL",  # = _nlopt.GN_DIRECT_L_RAND_NOSCAL
    "GN_ORIG_DIRECT",  # = _nlopt.GN_ORIG_DIRECT
    "GN_ORIG_DIRECT_L",  # = _nlopt.GN_ORIG_DIRECT_L
    "GD_STOGO",  # = _nlopt.GD_STOGO
    "GD_STOGO_RAND",  # = _nlopt.GD_STOGO_RAND
    "LD_LBFGS_NOCEDAL",  # = _nlopt.LD_LBFGS_NOCEDAL
    "LD_LBFGS",  # = _nlopt.LD_LBFGS
    "LN_PRAXIS",  # = _nlopt.LN_PRAXIS
    "LD_VAR1",  # = _nlopt.LD_VAR1
    "LD_VAR2",  # = _nlopt.LD_VAR2
    "LD_TNEWTON",  # = _nlopt.LD_TNEWTON
    "LD_TNEWTON_RESTART",  # = _nlopt.LD_TNEWTON_RESTART
    "LD_TNEWTON_PRECOND",  # = _nlopt.LD_TNEWTON_PRECOND
    "LD_TNEWTON_PRECOND_RESTART",  # = _nlopt.LD_TNEWTON_PRECOND_RESTART
    "GN_CRS2_LM",  # = _nlopt.GN_CRS2_LM
    "GN_MLSL",  # = _nlopt.GN_MLSL
    "GD_MLSL",  # = _nlopt.GD_MLSL
    "GN_MLSL_LDS",  # = _nlopt.GN_MLSL_LDS
    "GD_MLSL_LDS",  # = _nlopt.GD_MLSL_LDS
    "LD_MMA",  # = _nlopt.LD_MMA
    "LN_COBYLA",  # = _nlopt.LN_COBYLA
    "LN_NEWUOA",  # = _nlopt.LN_NEWUOA
    "LN_NEWUOA_BOUND",  # = _nlopt.LN_NEWUOA_BOUND
    "LN_NELDERMEAD",  # = _nlopt.LN_NELDERMEAD
    "LN_SBPLX",  # = _nlopt.LN_SBPLX
    "LN_AUGLAG",  # = _nlopt.LN_AUGLAG
    "LD_AUGLAG",  # = _nlopt.LD_AUGLAG
    "LN_AUGLAG_EQ",  # = _nlopt.LN_AUGLAG_EQ
    "LD_AUGLAG_EQ",  # = _nlopt.LD_AUGLAG_EQ
    "LN_BOBYQA",  # = _nlopt.LN_BOBYQA
    "GN_ISRES",  # = _nlopt.GN_ISRES
    "AUGLAG",  # = _nlopt.AUGLAG
    "AUGLAG_EQ",  # = _nlopt.AUGLAG_EQ
    "G_MLSL",  # = _nlopt.G_MLSL
    "G_MLSL_LDS",  # = _nlopt.G_MLSL_LDS
    "LD_SLSQP",  # = _nlopt.LD_SLSQP
    "LD_CCSAQ",  # = _nlopt.LD_CCSAQ
    "GN_ESCH",  # = _nlopt.GN_ESCH
    "GN_AGS",  # = _nlopt.GN_AGS
]

def _ensure_float(x):
    if isinstance(x, torch.Tensor): return float(x.detach().cpu().item())
    if isinstance(x, np.ndarray): return float(x.item())
    return float(x)

def _ensure_tensor(x):
    try:
        if isinstance(x, np.ndarray): return torch.as_tensor(x.copy())
    except SystemError:
        return None
    return torch.tensor(x, dtype=torch.float32)

inf = float('inf')
Closure = Callable[[bool], Any]

class NLOptWrapper(WrapperBase):
    """Use nlopt as pytorch optimizer, with gradient supplied by pytorch autograd.
    Note that this performs full minimization on each step,
    so usually you would want to perform a single step, although performing multiple steps will refine the
    solution.

    Args:
        params (Iterable): iterable of parameters to optimize or dicts defining parameter groups.
        algorithm (int | _ALGOS_LITERAL): optimization algorithm from https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
        maxeval (int | None):
            maximum allowed function evaluations, set to None to disable. But some stopping criterion
            must be set otherwise nlopt will run forever.
        lb (float | None, optional): optional lower bounds, some algorithms require this. Defaults to None.
        ub (float | None, optional): optional upper bounds, some algorithms require this. Defaults to None.
        stopval (float | None, optional): stop minimizing when an objective value ≤ stopval is found. Defaults to None.
        ftol_rel (float | None, optional): set relative tolerance on function value. Defaults to None.
        ftol_abs (float | None, optional): set absolute tolerance on function value. Defaults to None.
        xtol_rel (float | None, optional): set relative tolerance on optimization parameters. Defaults to None.
        xtol_abs (float | None, optional): set absolute tolerances on optimization parameters. Defaults to None.
        maxtime (float | None, optional): stop when the optimization time (in seconds) exceeds maxtime. Defaults to None.
    """
    def __init__(
        self,
        params,
        algorithm: int | _ALGOS_LITERAL,
        lb: float | None = None,
        ub: float | None = None,
        maxeval: int | None = None, # None can stall on some algos and because they are threaded C you can't even interrupt them
        stopval: float | None = None,
        ftol_rel: float | None = None,
        ftol_abs: float | None = None,
        xtol_rel: float | None = None,
        xtol_abs: float | None = None,
        maxtime: float | None = None,
        require_criterion: bool = True,
    ):
        if require_criterion:
            if all(i is None for i in (maxeval, stopval, ftol_abs, ftol_rel, xtol_abs, xtol_rel)):
                raise RuntimeError(
                    "Specify at least one stopping criterion out of "
                    "(maxeval, stopval, ftol_rel, ftol_abs, xtol_rel, xtol_abs, maxtime). "
                    "Pass `require_criterion=False` to suppress this error."
                )

        defaults = dict(lb=lb, ub=ub)
        super().__init__(params, defaults)

        self.opt: nlopt.opt | None = None
        self.algorithm_name: str | int = algorithm
        if isinstance(algorithm, str): algorithm = getattr(nlopt, algorithm.upper())
        self.algorithm: int = algorithm # type:ignore

        self.maxeval = maxeval; self.stopval = stopval
        self.ftol_rel = ftol_rel; self.ftol_abs = ftol_abs
        self.xtol_rel = xtol_rel; self.xtol_abs = xtol_abs
        self.maxtime = maxtime

        self._last_loss = None

    def _objective(self, x: np.ndarray, grad: np.ndarray, closure, params: TensorList):
        if self.raised:
            if self.opt is not None: self.opt.force_stop()
            return np.inf
        try:
            t = _ensure_tensor(x)
            if t is None:
                if self.opt is not None: self.opt.force_stop()
                return None
            params.from_vec_(t.to(params[0], copy=False))
            if grad.size > 0:
                with torch.enable_grad(): loss = closure()
                self._last_loss = _ensure_float(loss)
                grad[:] = params.grad.fill_none_(reference=params).to_vec().reshape(grad.shape).numpy(force=True)
                return self._last_loss

            self._last_loss = _ensure_float(closure(False))
            return self._last_loss
        except Exception as e:
            self.e = e
            self.raised = True
            if self.opt is not None: self.opt.force_stop()
            return np.inf

    @torch.no_grad
    def step(self, closure: Closure): # pylint: disable = signature-differs # pyright:ignore[reportIncompatibleMethodOverride]
        self.e = None
        self.raised = False
        params = TensorList(self._get_params())
        x0 = params.to_vec().numpy(force=True)

        plb, pub = self._get_per_parameter_lb_ub()
        if all(i is None for i in plb) and all(i is None for i in pub):
            lb = ub = None
        else:
            lb, ub = self._get_lb_ub(ld = {None: -np.inf}, ud = {None: np.inf})

        self.opt = nlopt.opt(self.algorithm, x0.size)
        self.opt.set_exceptions_enabled(False) # required
        self.opt.set_min_objective(partial(self._objective, closure = closure, params = params))
        if lb is not None: self.opt.set_lower_bounds(np.asarray(lb, dtype=x0.dtype))
        if ub is not None: self.opt.set_upper_bounds(np.asarray(ub, dtype=x0.dtype))

        if self.maxeval is not None: self.opt.set_maxeval(self.maxeval)
        if self.stopval is not None: self.opt.set_stopval(self.stopval)
        if self.ftol_rel is not None: self.opt.set_ftol_rel(self.ftol_rel)
        if self.ftol_abs is not None: self.opt.set_ftol_abs(self.ftol_abs)
        if self.xtol_rel is not None: self.opt.set_xtol_rel(self.xtol_rel)
        if self.xtol_abs is not None: self.opt.set_xtol_abs(self.xtol_abs)
        if self.maxtime is not None: self.opt.set_maxtime(self.maxtime)

        self._last_loss = None
        x = None
        try:
            x = self.opt.optimize(x0)
        # except SystemError as s:
        #     warnings.warn(f"{self.algorithm_name} raised {s}")
        except Exception as e:
            raise e from None

        if x is not None: params.from_vec_(torch.as_tensor(x, device = params[0].device, dtype=params[0].dtype))
        if self.e is not None: raise self.e from None

        if self._last_loss is None or x is None: return closure(False)

        return self._last_loss