# pylint:disable=not-callable
from collections.abc import Callable

import torch

from ...core import Chainable, Module
from ...utils import TensorList, vec_to_tensors
from ...utils.linalg.linear_operator import LinearOperator

from .trust_region import TrustRegionBase, _update_tr_radius

def _flatten_tensors(tensors: list[torch.Tensor]):
    return torch.cat([t.ravel() for t in tensors])

# code from https://github.com/konstmish/opt_methods/blob/master/optmethods/second_order/cubic.py
# ported to pytorch
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
        boundary_tol (float | None, optional):
            The trust region only increases when suggested step's norm is at least `(1-boundary_tol)*trust_region`.
            This prevents increasing trust region when solution is not on the boundary. Defaults to 1e-2.
        fallback (bool, optional):
            if ``True``, when ``hess_module`` maintains hessian inverse which can't be inverted efficiently, it will
            be inverted anyway. When ``False`` (default), a ``RuntimeError`` will be raised instead.
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
        nplus: float = 3.5,
        nminus: float = 0.25,
        rho_good: float = 0.99,
        rho_bad: float = 1e-4,
        init: float = 1,
        maxiter: int = 100,
        eps: float = 1e-8,
        update_freq: int = 1,
        max_attempts: int = 10,
        fallback: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = dict(init=init, nplus=nplus, nminus=nminus, rho_good=rho_good, rho_bad=rho_bad, eta=eta, maxiter=maxiter, eps=eps, max_attempts=max_attempts)
        super().__init__(hess_module=hess_module, requires="B", defaults=defaults, update_freq=update_freq, inner=inner, fallback=fallback)

    @torch.no_grad
    def trust_region_apply(self, var, tensors, B, H):
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
        if loss is None: loss = var.get_loss(False)

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
                trust_region=trust_region, settings = settings, boundary_fn=None,
            )

        assert d is not None
        if success: var.update = vec_to_tensors(d, params)
        else: var.update = params.zeros_like()

        return var