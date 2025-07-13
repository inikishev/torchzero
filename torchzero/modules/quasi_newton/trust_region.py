"""Trust region API is currently experimental, it will probably change completely"""
# pylint:disable=not-callable
from abc import ABC, abstractmethod
from typing import Any, Literal, cast, final
from collections.abc import Sequence, Mapping

import numpy as np
import torch
from scipy.optimize import lsq_linear

from ...core import Chainable, Module, apply_transform, Var
from ...utils import TensorList, vec_to_tensors
from ...utils.derivatives import (
    hessian_list_to_mat,
    jacobian_and_hessian_wrt,
)
from .quasi_newton import HessianUpdateStrategy
from ...utils.linalg import steihaug_toint_cg


def trust_lstsq(H: torch.Tensor, g: torch.Tensor, trust_region: float):
    res = lsq_linear(H.numpy(force=True).astype(np.float64), g.numpy(force=True).astype(np.float64), bounds=(-trust_region, trust_region))
    x = torch.from_numpy(res.x).to(H)
    return x, res.cost

def _flatten_tensors(tensors: list[torch.Tensor]):
    return torch.cat([t.ravel() for t in tensors])


class TrustRegionBase(Module, ABC):
    def __init__(
        self,
        defaults: dict | None = None,
        hess_module: HessianUpdateStrategy | None = None,
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):
        self._update_freq = update_freq
        super().__init__(defaults)

        if hess_module is not None:
            self.set_child('hess_module', hess_module)

        if inner is not None:
            self.set_child('inner', inner)

    @abstractmethod
    def trust_region_step(self, var: Var, tensors:list[torch.Tensor], P: torch.Tensor, is_inverse:bool) -> Var:
        """trust region logic"""


    @final
    @torch.no_grad
    def update(self, var):
        # ---------------------------------- update ---------------------------------- #
        closure = var.closure
        if closure is None: raise RuntimeError("Trust region requires closure")
        params = var.params

        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        P = None
        is_inverse=None
        g_list = var.grad
        loss = var.loss
        if step % self._update_freq == 0:

            if 'hess_module' not in self.children:
                params=var.params
                closure=var.closure
                if closure is None: raise ValueError('Closure is required for trust region')
                with torch.enable_grad():
                    loss = var.loss = var.loss_approx = closure(False)
                    g_list, H_list = jacobian_and_hessian_wrt([loss], params, batched=True)
                    g_list = [t[0] for t in g_list] # remove leading dim from loss
                    var.grad = g_list
                    P = hessian_list_to_mat(H_list)
                    is_inverse=False


            else:
                hessian_module = cast(HessianUpdateStrategy, self.children['hess_module'])
                hessian_module.update(var)
                P, is_inverse = hessian_module.get_B()

            if self._update_freq != 0:
                self.global_state['B'] = P
                self.global_state['is_inverse'] = is_inverse


    @final
    @torch.no_grad
    def apply(self, var):
        P = self.global_state['B']
        is_inverse = self.global_state['is_inverse']

        # -------------------------------- inner step -------------------------------- #
        update = var.get_update()
        if 'inner' in self.children:
            update = apply_transform(self.children['inner'], update, params=var.params, grads=var.grad, var=var)

        # ----------------------------------- apply ---------------------------------- #
        return self.trust_region_step(var=var, tensors=update, P=P, is_inverse=is_inverse)

def _update_tr_radius(update_vec:torch.Tensor, params: Sequence[torch.Tensor], closure,
                      loss, g:torch.Tensor, H:torch.Tensor, trust_region:float, settings: Mapping):
    """returns (update, new_trust_region)

    Args:
        update_vec (torch.Tensor): update vector which is SUBTRACTED from parameters
        params (_type_): params tensor list
        closure (_type_): closure
        loss (_type_): loss at x0
        g (torch.Tensor): gradient vector
        H (torch.Tensor): hessian
        trust_region (float): current trust region value
    """
    # evaluate actual loss reduction
    update_unflattned = vec_to_tensors(update_vec, params)
    params = TensorList(params)
    params -= update_unflattned
    loss_star = closure(False)
    params += update_unflattned
    reduction = loss - loss_star

    # expected reduction is g.T @ p + 0.5 * p.T @ B @ p
    if H.ndim == 1: Hu = H * update_vec
    else: Hu = H @ update_vec
    pred_reduction = - (g.dot(update_vec) + 0.5 * update_vec.dot(Hu))
    rho = reduction / (pred_reduction.clip(min=1e-8))

    # failed step
    if rho < 0.25:
        trust_region *= settings["nminus"]

    # very good step
    elif rho > 0.75:
        diff = trust_region - update_vec.abs()
        if (diff.amin() / trust_region) > 1e-4: # hits boundary
            trust_region *= settings["nplus"]

    # # if the ratio is high enough then accept the proposed step
    # if rho > settings["eta"]:
    #     update = vec_to_tensors(update_vec, params)

    # else:
    #     update = params.zeros_like()

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
        hess_module: HessianUpdateStrategy | None,
        eta: float= 0.15,
        nplus: float = 2,
        nminus: float = 0.25,
        init: float = 1,
        update_freq: int = 1,
        reg: float = 0,
        max_attempts: int = 10,
        inner: Chainable | None = None,
    ):
        defaults = dict(init=init, nplus=nplus, nminus=nminus, eta=eta, reg=reg, max_attempts=max_attempts)
        super().__init__(defaults, hess_module=hess_module, update_freq=update_freq, inner=inner)

    @torch.no_grad
    def trust_region_step(self, var, tensors, P, is_inverse):
        params = TensorList(var.params)
        settings = self.settings[params[0]]
        g = _flatten_tensors(tensors)

        reg = settings['reg']
        max_attempts = settings['max_attempts']

        loss = var.loss
        closure = var.closure
        if closure is None: raise RuntimeError("Trust region requires closure")
        if loss is None: loss = closure(False)

        if is_inverse:
            if P.ndim == 1: P = P.reciprocal()
            else: raise NotImplementedError()

        success = False
        update_vec = None
        while not success:
            max_attempts -= 1
            if max_attempts < 0: break

            trust_region = self.global_state.get('trust_region', settings['init'])

            if trust_region < 1e-8 or trust_region > 1e8:
                trust_region = self.global_state['trust_region'] = settings['init']

            update_vec = steihaug_toint_cg(P, g, trust_region, reg=reg)

            self.global_state['trust_region'], success = _update_tr_radius(
                update_vec=update_vec, params=params, closure=closure,
                loss=loss, g=g, H=P, trust_region=trust_region, settings = settings,
            )

        assert update_vec is not None
        if success: var.update = vec_to_tensors(update_vec, params)
        else: var.update = params.zeros_like()

        return var


# code from https://github.com/konstmish/opt_methods/blob/master/optmethods/second_order/cubic.py
# ported to torch
def ls_cubic_solver(f, g:torch.Tensor, H:torch.Tensor, M: float, is_inverse: bool, loss_plus, it_max=100, epsilon=1e-8, ):
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
    if is_inverse:
        newton_step = - H @ g
        H = torch.linalg.inv(H)
    else:
        newton_step, info = torch.linalg.solve_ex(H, g)
        if info != 0:
            newton_step = torch.linalg.lstsq(H, g).solution
        newton_step.neg_()
    if M == 0:
        return newton_step, solver_it
    def cauchy_point(g, H, M):
        if torch.linalg.vector_norm(g) == 0 or M == 0:
            return 0 * g
        g_dir = g / torch.linalg.vector_norm(g)
        H_g_g = H @ g_dir @ g_dir
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
    r_min = torch.linalg.vector_norm(cauchy_point(g, H, M))

    if f > loss_plus(newton_step):
        return newton_step, solver_it

    r_max = torch.linalg.vector_norm(newton_step)
    if r_max - r_min < epsilon:
        return newton_step, solver_it
    id_matrix = torch.eye(g.size(0), device=g.device, dtype=g.dtype)
    s_lam = None
    for _ in range(it_max):
        r_try = (r_min + r_max) / 2
        lam = r_try * M
        s_lam = -torch.linalg.solve(H + lam*id_matrix, g)
        solver_it += 1
        crit = conv_criterion(s_lam, r_try)
        if np.abs(crit) < epsilon:
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
                tz.m.CubicRegularization(),
            )

    """
    def __init__(
        self,
        hess_module: HessianUpdateStrategy | None = None,
        eta: float= 0.0,
        nplus: float = 2,
        nminus: float = 0.25,
        init: float = 1,
        maxiter: int = 100,
        eps: float = 1e-8,
        update_freq: int = 1,
        max_attempts: int = 10,
        inner: Chainable | None = None,
    ):
        defaults = dict(init=init, nplus=nplus, nminus=nminus, eta=eta, maxiter=maxiter, eps=eps, max_attempts=max_attempts)
        super().__init__(defaults, hess_module=hess_module, update_freq=update_freq, inner=inner)

    @torch.no_grad
    def trust_region_step(self, var, tensors, P, is_inverse):
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
        update_vec = None
        while not success:
            max_attempts -= 1
            if max_attempts < 0: break

            trust_region = self.global_state.get('trust_region', settings['init'])
            if trust_region < 1e-8 or trust_region > 1e16: trust_region = self.global_state['trust_region'] = settings['init']

            update_vec, _ = ls_cubic_solver(f=loss, g=g, H=P, M=1/trust_region, is_inverse=is_inverse,
                                    loss_plus=loss_plus, it_max=maxiter, epsilon=eps)
            update_vec.neg_()

            self.global_state['trust_region'], success = _update_tr_radius(
                update_vec=update_vec, params=params, closure=closure,
                loss=loss, g=g, H=P, trust_region=trust_region, settings = settings,
            )

        assert update_vec is not None
        if success: var.update = vec_to_tensors(update_vec, params)
        else: var.update = params.zeros_like()

        return var