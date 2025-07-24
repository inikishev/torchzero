import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast, final

import torch

from ...core import Chainable, Module, Var, apply_transform
from ...utils import TensorList, vec_to_tensors
from ...utils.linalg.linear_operator import LinearOperator


def _flatten_tensors(tensors: list[torch.Tensor]):
    return torch.cat([t.ravel() for t in tensors])

class TrustRegionBase(Module, ABC):
    def __init__(
        self,
        hess_module: Module,
        requires: Literal["B", "H", "any"],
        defaults: dict | None = None,
        fallback: bool = False,
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):
        self._update_freq = update_freq
        self._requires = requires
        self._fallback = fallback
        super().__init__(defaults)

        self.set_child('hess_module', hess_module)

        if inner is not None:
            self.set_child('inner', inner)


    def trust_region_update(self, var: Var, B: LinearOperator | None, H: LinearOperator | None) -> None:
        """updates the state of this module after H or B have been updated, if necessary"""

    @abstractmethod
    def trust_region_apply(self, var: Var, tensors:list[torch.Tensor], B: LinearOperator | None, H: LinearOperator | None) -> Var:
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
            B = H = None

            if self._requires == "B":
                B = hessian_module.get_B(var)
                if B is None:
                    if self._fallback:
                        H = hessian_module.get_H(var)
                        if H is None: raise RuntimeError(f"{hessian_module} doesn't support trust region")
                        B = H.inv()
                    else:
                        raise RuntimeError(f"{hessian_module} doesn't store hessian or hessian approximaton B")

            if self._requires == "H":
                H = hessian_module.get_H(var)
                if H is None:
                    if self._fallback:
                        B = hessian_module.get_B(var)
                        if B is None: raise RuntimeError(f"{hessian_module} doesn't support trust region")
                        H = B.inv()
                    else:
                        raise RuntimeError(f"{hessian_module} doesn't store hessian or hessian approximaton B")

            if self._requires == "any":
                H = hessian_module.get_H(var)
                B = hessian_module.get_B(var)

            self.global_state['B'] = B
            self.global_state["H"] = H

            self.trust_region_update(var, B=B, H=H)

            # if 'hess_module' not in self.children:
            #     params=var.params
            #     closure=var.closure
            #     if closure is None: raise ValueError('Closure is required for trust region')
            #     with torch.enable_grad():
            #         loss = var.loss = var.loss_approx = closure(False)
            #         g_list, H_list = jacobian_and_hessian_wrt([loss], params, batched=True)
            #         g_list = [t[0] for t in g_list] # remove leading dim from loss
            #         var.grad = g_list
            #         P = flatten_jacobian(H_list)
            #         is_inverse=False


            # else:
            #     hessian_module = cast(HessianUpdateStrategy, self.children['hess_module'])
            #     hessian_module.update(var)
            #     P, is_inverse = hessian_module.get_B()

            # if self._update_freq != 0:
            #     self.global_state['B'] = P
            #     self.global_state['is_inverse'] = is_inverse


    @final
    @torch.no_grad
    def apply(self, var):
        B = self.global_state.get('B', None)
        H = self.global_state.get('H', None)

        # -------------------------------- inner step -------------------------------- #
        update = var.get_update()
        if 'inner' in self.children:
            update = apply_transform(self.children['inner'], update, params=var.params, grads=var.grad, var=var)

        # ----------------------------------- apply ---------------------------------- #
        return self.trust_region_apply(var=var, tensors=update, B=B, H=H)

def _l2_boundary_check(d: torch.Tensor, trust_region: float, boundary_tol: float | None):
    if boundary_tol is None: return True
    magn = torch.linalg.vector_norm(d) # pylint:disable=not-callable
    return (trust_region - magn) / trust_region < boundary_tol

def _update_tr_radius(params: Sequence[torch.Tensor], closure,
                      d:torch.Tensor, f, g:torch.Tensor, H: LinearOperator | None, B:LinearOperator | None,
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
        H (LinearOperator | None): hessian inverse approximation (currently not used).
        B (LinearOperator | None): hessian approximation
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

    if B is not None:
        # expected reduction is g.T @ p + 0.5 * p.T @ B @ p
        Hu = B.matvec(d)
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
    else: d_region = boundary_fn(trust_region)

    # failed step
    if rho < settings['rho_bad'] or not is_finite:
        trust_region = d_region*settings["nminus"]

    # very good step
    elif rho > settings['rho_good'] and is_finite:
        boundary_tol=settings["boundary_tol"]
        if (boundary_tol is None) or (trust_region-d_region)/trust_region < boundary_tol:
            trust_region = max(trust_region, d_region*settings["nplus"])

    # return new trust region and success boolean
    return trust_region, rho > settings["eta"] and is_finite
