"""Trust region API is currently experimental, it will probably change completely"""
# pylint:disable=not-callable
from abc import ABC, abstractmethod
from typing import Any, Literal, cast, final
from collections.abc import Sequence, Mapping, Callable

import numpy as np
import torch
from scipy.optimize import lsq_linear

from ...core import Chainable, Module, apply_transform, Var
from ...utils import TensorList, vec_to_tensors
from ...utils.derivatives import (
    flatten_jacobian,
    jacobian_and_hessian_wrt,
)
from .quasi_newton import HessianUpdateStrategy
from ...utils.linalg import cg
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

    @abstractmethod
    def trust_region_step(self, var: Var, tensors:list[torch.Tensor], B: LinearOperator | None, H: LinearOperator | None) -> Var:
        """trust region logic"""


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
        return self.trust_region_step(var=var, tensors=update, B=B, H=H)

def _update_tr_radius(params: Sequence[torch.Tensor], closure,
                      d:torch.Tensor, f, g:torch.Tensor, H: LinearOperator | None, B:LinearOperator | None,
                      trust_region:float, settings: Mapping):
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
        H (LinearOperator | None): hessian inverse approximation.
        B (LinearOperator | None): hessian approximation
        trust_region (float): current trust region value
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

    # failed step
    if rho < 0.25:
        trust_region *= settings["nminus"]

    # very good step
    elif rho > 0.75:
        magn = torch.linalg.vector_norm(d) # pylint:disable=not-callable
        if settings['boundary_tol'] is None or (magn - trust_region) / trust_region > -settings['boundary_tol']: # close to boundary
            trust_region *= settings["nplus"]

    return trust_region, rho > settings["eta"]

class TrustCG(TrustRegionBase):
    """Trust region via Steihaug-Toint Conjugate Gradient method. This is mainly useful for quasi-newton methods.
    If you don't use :code:`hess_module`, use the matrix-free :code:`tz.m.NewtonCGSteihaug` which only uses hessian-vector products.

    Args:
        hess_module (HessianUpdateStrategy | None, optional):
            Hessian update strategy, must be one of the :code:`HessianUpdateStrategy` modules. Make sure to set :code:`inverse=False`. If None, uses autograd to calculate the hessian. Defaults to None.
        eta (float, optional):
            if ratio of actual to predicted rediction is larger than this, step is accepted.
            When :code:`hess_module` is None, this can be set to 0. Defaults to 0.15.
        nplus (float, optional): increase factor on successful steps. Defaults to 1.5.
        nminus (float, optional): decrease factor on unsuccessful steps. Defaults to 0.75.
        init (float, optional): Initial trust region value. Defaults to 1.
        update_freq (int, optional): frequency of updating the hessian. Defaults to 1.
        reg (int, optional): hessian regularization. Defaults to 0.
        inner (Chainable | None, optional): preconditioning is applied to output of thise module. Defaults to None.

    Examples:
        Trust-SR1

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.TrustCG(hess_module=tz.m.SR1(inverse=False)),
            )
    """
    def __init__(
        self,
        hess_module: Module,
        eta: float= 0.15,
        nplus: float = 2,
        nminus: float = 0.25,
        init: float = 1,
        update_freq: int = 1,
        reg: float = 0,
        max_attempts: int = 10,
        boundary_tol: float | None = 1e-2,
        fallback: bool = False,
        inner: Chainable | None = None,
    ):
        defaults = dict(init=init, nplus=nplus, nminus=nminus, eta=eta, reg=reg, max_attempts=max_attempts,boundary_tol=boundary_tol)
        super().__init__(hess_module=hess_module, requires="B", defaults=defaults, update_freq=update_freq, inner=inner, fallback=fallback)

    @torch.no_grad
    def trust_region_step(self, var, tensors, B, H):
        assert B is not None

        params = TensorList(var.params)
        settings = self.settings[params[0]]
        g = _flatten_tensors(tensors)

        reg = settings['reg']
        max_attempts = settings['max_attempts']

        loss = var.loss
        closure = var.closure
        if closure is None: raise RuntimeError("Trust region requires closure")
        if loss is None: loss = closure(False)

        success = False
        d = None
        while not success:
            max_attempts -= 1
            if max_attempts < 0: break

            trust_region = self.global_state.get('trust_region', settings['init'])

            if trust_region < 1e-8 or trust_region > 1e8:
                trust_region = self.global_state['trust_region'] = settings['init']

            d = cg(B.matvec, g, trust_region=trust_region, reg=reg)

            self.global_state['trust_region'], success = _update_tr_radius(
                params=params, closure=closure, d=d, f=loss, g=g, B=B, H=None,
                trust_region=trust_region, settings = settings,
            )

        assert d is not None
        if success: var.update = vec_to_tensors(d, params)
        else: var.update = params.zeros_like()

        return var


# code from https://github.com/konstmish/opt_methods/blob/master/optmethods/second_order/cubic.py
# ported to torch
def ls_cubic_solver(f, g:torch.Tensor, B:LinearOperator, M: float, loss_plus: Callable, it_max=100, epsilon=1e-8, ):
    """
    Solve min_z <g, z-x> + 1/2<z-x, H(z-x)> + M/3 ||z-x||^3

    For explanation of Cauchy point, see "Gradient Descent
        Efficiently Finds the Cubic-Regularized Non-Convex Newton Step"
        https://arxiv.org/pdf/1612.00547.pdf
    Other potential implementations can be found in paper
        "Adaptive cubic regularisation methods"
        https://people.maths.ox.ac.uk/cartis/papers/ARCpI.pdf
    """
    solver_it = 1
    newton_step = B.solve(g).neg_()
    if M == 0:
        return newton_step, solver_it

    def cauchy_point(g, B:LinearOperator, M):
        if torch.linalg.vector_norm(g) == 0 or M == 0:
            return 0 * g
        g_dir = g / torch.linalg.vector_norm(g)
        H_g_g = B.matvec(g_dir) @ g_dir
        R = -H_g_g / (2*M) + torch.sqrt((H_g_g/M)**2/4 + torch.linalg.vector_norm(g)/M)
        return -R * g_dir

    def conv_criterion(s, r):
        """
        The convergence criterion is an increasing and concave function in r
        and it is equal to 0 only if r is the solution to the cubic problem
        """
        s_norm = torch.linalg.vector_norm(s)
        return 1/s_norm - 1/r

    # Solution s satisfies ||s|| >= Cauchy_radius
    r_min = torch.linalg.vector_norm(cauchy_point(g, B, M))

    if f > loss_plus(newton_step):
        return newton_step, solver_it

    r_max = torch.linalg.vector_norm(newton_step)
    if r_max - r_min < epsilon:
        return newton_step, solver_it
    # id_matrix = torch.eye(g.size(0), device=g.device, dtype=g.dtype)
    s_lam = None
    for _ in range(it_max):
        r_try = (r_min + r_max) / 2
        lam = r_try * M
        s_lam = B.add_diagonal(lam).solve(g).neg()
        # s_lam = -torch.linalg.solve(B + lam*id_matrix, g)
        solver_it += 1
        crit = conv_criterion(s_lam, r_try)
        if torch.abs(crit) < epsilon:
            return s_lam, solver_it
        if crit < 0:
            r_min = r_try
        else:
            r_max = r_try
        if r_max - r_min < epsilon:
            break
    assert s_lam is not None
    return s_lam, solver_it


class CubicRegularization(TrustRegionBase):
    """Cubic regularization.

    .. note::
        by default this functions like a trust region, set nplus and nminus = 1 to make regularization parameter fixed.
        :code:`init` sets 1/regularization.

    Args:
        hess_module (HessianUpdateStrategy | None, optional):
            Hessian update strategy, must be one of the :code:`HessianUpdateStrategy` modules. This works better with true hessian though. Make sure to set :code:`inverse=False`. If None, uses autograd to calculate the hessian. Defaults to None.
        eta (float, optional):
            if ratio of actual to predicted rediction is larger than this, step is accepted.
            When :code:`hess_module` is None, this can be set to 0. Defaults to 0.0.
        nplus (float, optional): increase factor on successful steps. Defaults to 1.5.
        nminus (float, optional): decrease factor on unsuccessful steps. Defaults to 0.75.
        init (float, optional): Initial trust region value. Defaults to 1.
        maxiter (float, optional): maximum iterations when solving cubic subproblem, defaults to 1e-7.
        eps (float, optional): epsilon for the solver, defaults to 1e-8.
        update_freq (int, optional): frequency of updating the hessian. Defaults to 1.
        inner (Chainable | None, optional): preconditioning is applied to output of thise module. Defaults to None.

    Examples:
        Cubic regularized newton

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.CubicRegularization(tz.m.Newton()),
            )

    """
    def __init__(
        self,
        hess_module: Module,
        eta: float= 0.0,
        nplus: float = 2,
        nminus: float = 0.25,
        init: float = 1,
        maxiter: int = 100,
        eps: float = 1e-8,
        update_freq: int = 1,
        max_attempts: int = 10,
        boundary_tol: float | None = None,
        fallback: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = dict(init=init, nplus=nplus, nminus=nminus, eta=eta, maxiter=maxiter, eps=eps, max_attempts=max_attempts, boundary_tol=boundary_tol)
        super().__init__(hess_module=hess_module, requires="B", defaults=defaults, update_freq=update_freq, inner=inner, fallback=fallback)

    @torch.no_grad
    def trust_region_step(self, var, tensors, B, H):
        assert B is not None # have to use B to calculate predicted reduction

        params = TensorList(var.params)
        settings = self.settings[params[0]]
        g = _flatten_tensors(tensors)

        maxiter = settings['maxiter']
        max_attempts = settings['max_attempts']
        eps = settings['eps']

        loss = var.loss
        closure = var.closure
        if closure is None: raise RuntimeError("Trust region requires closure")
        if loss is None: loss = closure(False)

        def loss_plus(x):
            x_unflat = vec_to_tensors(x, params)
            params.add_(x_unflat)
            loss_x = closure(False)
            params.sub_(x_unflat)
            return loss_x

        success = False
        d = None
        while not success:
            max_attempts -= 1
            if max_attempts < 0: break

            trust_region = self.global_state.get('trust_region', settings['init'])
            if trust_region < 1e-8 or trust_region > 1e16: trust_region = self.global_state['trust_region'] = settings['init']

            d, _ = ls_cubic_solver(f=loss, g=g, B=B, M=1/trust_region, loss_plus=loss_plus, it_max=maxiter, epsilon=eps)
            d.neg_()

            self.global_state['trust_region'], success = _update_tr_radius(
                params=params, closure=closure, d=d, f=loss, g=g, B=None, H=H,
                trust_region=trust_region, settings = settings,
            )

        assert d is not None
        if success: var.update = vec_to_tensors(d, params)
        else: var.update = params.zeros_like()

        return var