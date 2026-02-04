import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence, MutableMapping
from functools import partial
from typing import Any, Literal, Protocol, cast, final, overload

import torch

from ...core import Chainable, Module, Objective
from ...linalg.linear_operator import LinearOperator
from ...utils import (
    TensorList,
    generic_finfo,
    generic_vector_norm,
    safe_dict_update_,
    tofloat,
    vec_to_tensors,
)


def _flatten_tensors(tensors: list[torch.Tensor]):
    return torch.cat([t.ravel() for t in tensors])



class _RadiusStrategy(Protocol):
    def __call__(
        self,
        params: Sequence[torch.Tensor],
        closure: Callable,
        f: float,
        g: torch.Tensor,
        H: LinearOperator,
        d: torch.Tensor,
        trust_radius: float,
        eta: float, # 0.0
        nplus: float, # 3.5
        nminus: float, # 0.25
        rho_good: float, # 0.99
        rho_bad: float, # 1e-4
        boundary_tol: float | None,
        init: float,
        state: Mapping[str, Any],
        settings: Mapping[str, Any],
        radius_fn: Callable | None = torch.linalg.vector_norm,
    ) -> tuple[float, bool]:
        """returns (new trust_region value, success).

        Args:
            params (Sequence[torch.Tensor]): params tensor list
            closure (Callable): closure
            d (torch.Tensor):
                current update vector with current trust_region, which is SUBTRACTED from parameters.
                May be exact solution to (B+yI)x=g, approximate, or a solution to a different subproblem
                (e.g. cubic regularization).
            f (float | torch.Tensor): loss at x0
            g (torch.Tensor): gradient vector
            H (LinearOperator | None): hessian approximation
            trust_radius (float): current trust region value
            eta (float, optional):
                if ratio of actual to predicted rediction is larger than this, step is accepted.
                When :code:`hess_module` is GaussNewton, this can be set to 0. Defaults to 0.15.
            nplus (float, optional): increase factor on successful steps. Defaults to 1.5.
            nminus (float, optional): decrease factor on unsuccessful steps. Defaults to 0.75.
            rho_good (float, optional):
                if ratio of actual to predicted rediction is larger than this, trust region size is multiplied by `nplus`.
            rho_bad (float, optional):
                if ratio of actual to predicted rediction is less than this, trust region size is multiplied by `nminus`.
            boundary_tol (float | None, optional):
                The trust region only increases when suggested step's norm is at least `(1-boundary_tol)*trust_region`.
                This prevents increasing trust region when solution is not on the boundary. Defaults to 1e-2.
            init (float, optional): Initial trust region value. Defaults to 1.
            state (dict, optional): global state of the module for storing persistent info.
            settings (dict, optional): all settings in case this strategy has other settings.
            radius_fn (Callable | None, optional):
                function that accepts ``(d: torch.Tensor)`` and returns the actual region of ``d``
                (e.g. L2) norm for L2 trust region.
        """
        ... # pylint:disable=unnecessary-ellipsis

def _get_rho(params: Sequence[torch.Tensor], closure:Callable,
             f: float, g: torch.Tensor, H: LinearOperator, d:torch.Tensor, ):
    """rho is reduction/pred_reduction"""

    # evaluate actual loss reduction
    update_unflattned = vec_to_tensors(d, params)
    params = TensorList(params)
    x0 = params.clone() # same as in line searches, large directions are undone very imprecisely

    params -= update_unflattned
    f_star = closure(False)
    params.set_(x0)

    reduction = f - f_star

    # expected reduction is g.T @ p + 0.5 * p.T @ B @ p
    Hu = H.matvec(d)
    pred_reduction = g.dot(d) - 0.5 * d.dot(Hu)

    rho = reduction / (pred_reduction.clip(min=torch.finfo(g.dtype).tiny * 2))
    return rho, f_star, reduction, pred_reduction

def _get_rho_tensorlist(params: Sequence[torch.Tensor], closure:Callable,
             f: float, g: TensorList, Hvp: Callable[[TensorList], TensorList], d:TensorList):
    """rho is reduction/pred_reduction"""
    params = TensorList(params)
    x0 = params.clone() # same as in line searches, large directions are undone very imprecisely

    # evaluate before modifying params to not break autograd
    Hu = Hvp(d)

    # actual f
    params -= d
    f_star = closure(False)
    params.copy_(x0)

    reduction = f - f_star

    # expected f is g.T @ p + 0.5 * p.T @ B @ p
    pred_reduction = g.dot(d) - 0.5 * d.dot(Hu)

    rho = reduction / (pred_reduction.clip(min=torch.finfo(g[0].dtype).tiny * 2))
    return rho, f_star, reduction, pred_reduction

@torch.no_grad
def default_radius(
    params: Sequence[torch.Tensor],
    closure: Callable,
    f: float,
    g: torch.Tensor | TensorList,
    H: LinearOperator | Callable,
    d: torch.Tensor | TensorList,
    trust_radius: float,
    eta: float, # 0.0
    nplus: float, # 3.5
    nminus: float, # 0.25
    rho_good: float, # 0.99
    rho_bad: float, # 1e-4
    boundary_tol: float | None,
    init: float,
    state: Mapping[str, Any],
    settings: Mapping[str, Any],
    radius_fn: Callable | None = generic_vector_norm,
    check_overflow: bool = True,
    # dynamic_nminus: bool=False,
) -> tuple[float, bool]:

    # when rho_bad < rho < eta, no update is made but trust region is not updated.
    if eta > rho_bad:
        warnings.warn(f"trust region eta={eta} is larger than rho_bad={rho_bad}, "
                      "this can lead to trust region getting stuck.")

    if isinstance(g, torch.Tensor):
        rho, f_star, _, _ = _get_rho(params=params, closure=closure, f=f, g=g, H=H, d=d) # pyright:ignore[reportArgumentType]
    else:
        rho, f_star, _, _ = _get_rho_tensorlist(params=params, closure=closure, f=f, g=g, Hvp=H, d=d) # pyright:ignore[reportArgumentType]

    is_finite = math.isfinite(f_star)

    # find boundary of current step
    if radius_fn is None: d_radius = trust_radius
    else: d_radius = radius_fn(d)

    # failed step
    if rho < rho_bad or not is_finite:
        # if dynamic_nminus and rho > 0: nminus = nminus * max(rho, 1e-4)
        trust_radius = d_radius*nminus

    # very good step
    elif rho > rho_good and is_finite:
        if (boundary_tol is None) or (trust_radius-d_radius)/trust_radius < boundary_tol:
            trust_radius = max(trust_radius, d_radius*nplus)

    # prevent very small or large values
    if check_overflow:
        finfo = generic_finfo(g)
        if trust_radius < finfo.tiny*2 or trust_radius > finfo.max/2:
            trust_radius = init

    # return new trust region and success boolean
    return tofloat(trust_radius), rho > eta and is_finite


def fixed_radius(
    params: Sequence[torch.Tensor],
    closure: Callable,
    f: float,
    g: torch.Tensor,
    H: LinearOperator,
    d: torch.Tensor,
    trust_radius: float,
    eta: float, # 0.0
    nplus: float, # 3.5
    nminus: float, # 0.25
    rho_good: float, # 0.99
    rho_bad: float, # 1e-4
    boundary_tol: float | None,
    init: float,
    state: Mapping[str, Any],
    settings: Mapping[str, Any],
    radius_fn: Callable | None = torch.linalg.vector_norm,
) -> tuple[float, bool]:
    return init, True


_RADIUS_KEYS = Literal['default', 'fixed']
_RADIUS_STRATEGIES: dict[_RADIUS_KEYS, _RadiusStrategy] = {
    "default": default_radius,
    "fixed": fixed_radius,
    # "dynamic": partial(default_radius, dynamic_nminus=True)
}

class TrustRegionBase(Module, ABC):
    def __init__(
        self,
        defaults: dict | None,
        hess_module: Chainable,
        # suggested default values:
        # Gould, Nicholas IM, et al. "Sensitivity of trust-region algorithms to their parameters." 4OR 3.3 (2005): 227-241.
        # which I found from https://github.com/patrick-kidger/optimistix/blob/c1dad7e75fc35bd5a4977ac3a872991e51e83d2c/optimistix/_solver/trust_region.py#L113-200
        eta: float, # 0.0
        nplus: float, # 3.5
        nminus: float, # 0.25
        rho_good: float, # 0.99
        rho_bad: float, # 1e-4
        boundary_tol: float | None, # None or 1e-1
        init: float, # 1
        max_attempts: int, # 10
        radius_strategy: _RadiusStrategy | _RADIUS_KEYS, # "default"
        radius_fn: Callable | None, # torch.linalg.vector_norm
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):
        if isinstance(radius_strategy, str): radius_strategy = _RADIUS_STRATEGIES[radius_strategy]
        if defaults is None: defaults = {}

        safe_dict_update_(
            defaults,
            dict(eta=eta, nplus=nplus, nminus=nminus, rho_good=rho_good, rho_bad=rho_bad, init=init,
                 update_freq=update_freq, max_attempts=max_attempts, radius_strategy=radius_strategy,
                 boundary_tol=boundary_tol)
        )

        super().__init__(defaults)

        self._radius_fn = radius_fn
        self.set_child('hess_module', hess_module)

        if inner is not None:
            self.set_child('inner', inner)

    @abstractmethod
    def trust_solve(
        self,
        f: float,
        g: torch.Tensor,
        H: LinearOperator,
        radius: float,
        params: list[torch.Tensor],
        closure: Callable,
        settings: Mapping[str, Any],
    ) -> torch.Tensor:
        """Solve Hx=g with a trust region penalty/bound defined by `radius`"""
        ... # pylint:disable=unnecessary-ellipsis

    def trust_region_update(self, objective: Objective, H: LinearOperator | None) -> None:
        """updates the state of this module after H or B have been updated, if necessary"""

    def trust_region_apply(self, objective: Objective, tensors:list[torch.Tensor], H: LinearOperator | None) -> Objective:
        """Solves the trust region subproblem and outputs ``Objective`` with the solution direction."""
        assert H is not None

        params = TensorList(objective.params)
        settings = self.settings[params[0]]
        g = _flatten_tensors(tensors)

        max_attempts = settings['max_attempts']

        # loss at x_0
        loss = objective.loss
        closure = objective.closure
        if closure is None: raise RuntimeError("Trust region requires closure")
        if loss is None: loss = objective.get_loss(False)
        loss = tofloat(loss)

        # trust region step and update
        success = False
        d = None
        while not success:
            max_attempts -= 1
            if max_attempts < 0: break

            trust_radius = self.global_state.get('trust_radius', settings['init'])

            # solve Hx=g
            d = self.trust_solve(f=loss, g=g, H=H, radius=trust_radius, params=params, closure=closure, settings=settings)

            # update trust radius
            radius_strategy: _RadiusStrategy = settings['radius_strategy']
            self.global_state["trust_radius"], success = radius_strategy(
                params=params,
                closure=closure,
                d=d,
                f=loss,
                g=g,
                H=H,
                trust_radius=trust_radius,

                eta=settings["eta"],
                nplus=settings["nplus"],
                nminus=settings["nminus"],
                rho_good=settings["rho_good"],
                rho_bad=settings["rho_bad"],
                boundary_tol=settings["boundary_tol"],
                init=settings["init"],

                state=self.global_state,
                settings=settings,
                radius_fn=self._radius_fn,
            )

        assert d is not None
        if success: objective.updates = vec_to_tensors(d, params)
        else: objective.updates = params.zeros_like()

        return objective


    @final
    @torch.no_grad
    def update(self, objective):
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        if step % self.defaults["update_freq"] == 0:

            hessian_module = self.children['hess_module']
            hessian_module.update(objective)
            H = hessian_module.get_H(objective)
            self.global_state["H"] = H

            self.trust_region_update(objective, H=H)


    @final
    @torch.no_grad
    def apply(self, objective):
        H = self.global_state.get('H', None)

        # -------------------------------- inner step -------------------------------- #
        objective = self.inner_step("inner", objective, must_exist=False)

        # ----------------------------------- apply ---------------------------------- #
        return self.trust_region_apply(objective=objective, tensors=objective.get_updates(), H=H)

