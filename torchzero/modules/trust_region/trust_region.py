import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast, final

import torch

from ...core import Chainable, Module, Var, apply_transform
from ...utils import TensorList, vec_to_tensors, tofloat
from ...utils.linalg.linear_operator import LinearOperator


def _flatten_tensors(tensors: list[torch.Tensor]):
    return torch.cat([t.ravel() for t in tensors])

class TrustRegionBase(Module, ABC):
    def __init__(
        self,
        hess_module: Module,
        defaults: dict | None = None,
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):
        self._update_freq = update_freq
        super().__init__(defaults)

        self.set_child('hess_module', hess_module)

        if inner is not None:
            self.set_child('inner', inner)


    def trust_region_update(self, var: Var, H: LinearOperator | None) -> None:
        """updates the state of this module after H or B have been updated, if necessary"""

    @abstractmethod
    def trust_region_apply(self, var: Var, tensors:list[torch.Tensor], H: LinearOperator | None) -> Var:
        """Solves the trust region subproblem and outputs ``Var`` with the solution direction."""

    @final
    @torch.no_grad
    def update(self, var):
        # ---------------------------------- update ---------------------------------- #
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        if step % self._update_freq == 0:

            hessian_module = self.children['hess_module']
            hessian_module.update(var)
            H = hessian_module.get_H(var)
            self.global_state["H"] = H

            self.trust_region_update(var, H=H)


    @final
    @torch.no_grad
    def apply(self, var):
        H = self.global_state.get('H', None)

        # -------------------------------- inner step -------------------------------- #
        update = var.get_update()
        if 'inner' in self.children:
            update = apply_transform(self.children['inner'], update, params=var.params, grads=var.grad, var=var)

        # ----------------------------------- apply ---------------------------------- #
        return self.trust_region_apply(var=var, tensors=update, H=H)


def _update_tr_radius(params: Sequence[torch.Tensor], closure,
                      d:torch.Tensor, f, g:torch.Tensor, H:LinearOperator | None,
                      trust_region:float, settings: Mapping, boundary_fn: Callable | None=torch.linalg.vector_norm,):
    """returns (new trust_region value, success). If B is not specified this depends on how accurate `d` is,
    so don't pass different subproblems.

    Args:
        params (Sequence[torch.Tensor]): params tensor list
        closure (Callable): closure
        d (torch.Tensor):
            current update vector with current trust_region, which is SUBTRACTED from parameters.
            May be exact solution to (B+yI)x=g, approximate, or a solution to a compeletely different subproblem
            (e.g. cubic regularization).
        f (float | torch.Tensor): loss at x0
        g (torch.Tensor): gradient vector
        H (LinearOperator | None): hessian approximation
        trust_region (float): current trust region value
        boundary_check (Callable | None, optional):
            function that accepts ``(d: torch.Tensor)`` and returns the actual region of ``d``
            (e.g. L2) norm for L2 trust region.
    """
    # evaluate actual loss reduction
    update_unflattned = vec_to_tensors(d, params)
    params = TensorList(params)
    params -= update_unflattned
    loss_star = closure(False)
    params += update_unflattned
    reduction = f - loss_star

    if H is not None:
        # expected reduction is g.T @ p + 0.5 * p.T @ B @ p
        Hu = H.matvec(d)
        pred_reduction = - (g.dot(d) + 0.5 * d.dot(Hu))

    else:
        # this may be less accurate? because it depends on how accurate `d` is
        # the formula (if d was not negative) is -0.5 * g^T d + 0.5 * λ * ||d||²
        # I will keep H in args in case there is a better method but I haven't found anything
        pred_reduction = 0.5 * trust_region * d.dot(d) + 0.5 * g.dot(d)

    rho = reduction / (pred_reduction.clip(min=1e-8))
    is_finite = math.isfinite(loss_star)

    # find boundary of current step
    if boundary_fn is None: d_region = trust_region
    else: d_region = boundary_fn(d)

    # failed step
    if rho < settings['rho_bad'] or not is_finite:
        trust_region = d_region*settings["nminus"]

    # very good step
    elif rho > settings['rho_good'] and is_finite:
        boundary_tol=settings.get("boundary_tol", None)
        if (boundary_tol is None) or (trust_region-d_region)/trust_region < boundary_tol:
            trust_region = max(trust_region, d_region*settings["nplus"])

    # return new trust region and success boolean
    return tofloat(trust_region), rho > settings["eta"] and is_finite
