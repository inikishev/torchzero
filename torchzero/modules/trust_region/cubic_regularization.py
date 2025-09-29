# pylint:disable=not-callable
from collections.abc import Callable

import torch

from ...core import Chainable, Module
from ...utils import TensorList, vec_to_tensors
from ...linalg.linear_operator import LinearOperator
from .trust_region import _RADIUS_KEYS, TrustRegionBase, _RadiusStrategy


# code from https://github.com/konstmish/opt_methods/blob/master/optmethods/second_order/cubic.py
# ported to pytorch and linear operator
def ls_cubic_solver(f, g:torch.Tensor, H:LinearOperator, M: float, loss_at_params_plus_x_fn: Callable | None, it_max=100, epsilon=1e-8, ):
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
    newton_step = H.solve(g).neg_()
    if M == 0:
        return newton_step, solver_it

    def cauchy_point(g, H:LinearOperator, M):
        if torch.linalg.vector_norm(g) == 0 or M == 0:
            return 0 * g
        g_dir = g / torch.linalg.vector_norm(g)
        H_g_g = H.matvec(g_dir) @ g_dir
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

    if (loss_at_params_plus_x_fn is not None) and (f > loss_at_params_plus_x_fn(newton_step)):
        return newton_step, solver_it

    r_max = torch.linalg.vector_norm(newton_step)
    if r_max - r_min < epsilon:
        return newton_step, solver_it

    # id_matrix = torch.eye(g.size(0), device=g.device, dtype=g.dtype)
    s_lam = None
    for _ in range(it_max):
        r_try = (r_min + r_max) / 2
        lam = r_try * M
        s_lam = H.solve_plus_diag(g, lam).neg()
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

    Args:
        hess_module (Module | None, optional):
            A module that maintains a hessian approximation (not hessian inverse!).
            This includes all full-matrix quasi-newton methods, ``tz.m.Newton`` and ``tz.m.GaussNewton``.
            When using quasi-newton methods, set `inverse=False` when constructing them.
        eta (float, optional):
            if ratio of actual to predicted rediction is larger than this, step is accepted.
            When :code:`hess_module` is GaussNewton, this can be set to 0. Defaults to 0.15.
        nplus (float, optional): increase factor on successful steps. Defaults to 1.5.
        nminus (float, optional): decrease factor on unsuccessful steps. Defaults to 0.75.
        rho_good (float, optional):
            if ratio of actual to predicted rediction is larger than this, trust region size is multiplied by `nplus`.
        rho_bad (float, optional):
            if ratio of actual to predicted rediction is less than this, trust region size is multiplied by `nminus`.
        init (float, optional): Initial trust region value. Defaults to 1.
        maxiter (float, optional): maximum iterations when solving cubic subproblem, defaults to 1e-7.
        eps (float, optional): epsilon for the solver, defaults to 1e-8.
        update_freq (int, optional): frequency of updating the hessian. Defaults to 1.
        max_attempts (max_attempts, optional):
            maximum number of trust region size size reductions per step. A zero update vector is returned when
            this limit is exceeded. Defaults to 10.
        fallback (bool, optional):
            if ``True``, when ``hess_module`` maintains hessian inverse which can't be inverted efficiently, it will
            be inverted anyway. When ``False`` (default), a ``RuntimeError`` will be raised instead.
        inner (Chainable | None, optional): preconditioning is applied to output of thise module. Defaults to None.


    Examples:
        Cubic regularized newton

        .. code-block:: python

            opt = tz.Optimizer(
                model.parameters(),
                tz.m.CubicRegularization(tz.m.Newton()),
            )

    """
    def __init__(
        self,
        hess_module: Chainable,
        eta: float= 0.0,
        nplus: float = 3.5,
        nminus: float = 0.25,
        rho_good: float = 0.99,
        rho_bad: float = 1e-4,
        init: float = 1,
        max_attempts: int = 10,
        radius_strategy: _RadiusStrategy | _RADIUS_KEYS = 'default',
        maxiter: int = 100,
        eps: float = 1e-8,
        check_decrease:bool=False,
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):
        defaults = dict(maxiter=maxiter, eps=eps, check_decrease=check_decrease)
        super().__init__(
            defaults=defaults,
            hess_module=hess_module,
            eta=eta,
            nplus=nplus,
            nminus=nminus,
            rho_good=rho_good,
            rho_bad=rho_bad,
            init=init,
            max_attempts=max_attempts,
            radius_strategy=radius_strategy,
            update_freq=update_freq,
            inner=inner,

            boundary_tol=None,
            radius_fn=None,
        )

    def trust_solve(self, f, g, H, radius, params, closure, settings):
        params = TensorList(params)

        loss_at_params_plus_x_fn = None
        if settings['check_decrease']:
            def closure_plus_x(x):
                x_unflat = vec_to_tensors(x, params)
                params.add_(x_unflat)
                loss_x = closure(False)
                params.sub_(x_unflat)
                return loss_x
            loss_at_params_plus_x_fn = closure_plus_x


        d, _ = ls_cubic_solver(f=f, g=g, H=H, M=1/radius, loss_at_params_plus_x_fn=loss_at_params_plus_x_fn,
                               it_max=settings['maxiter'], epsilon=settings['eps'])
        return d.neg_()
