import torch

from ...core import Chainable, Module
from ...linalg import cg, linear_operator
from .trust_region import _RADIUS_KEYS, TrustRegionBase, _RadiusStrategy


class TrustCG(TrustRegionBase):
    """Trust region via Steihaug-Toint Conjugate Gradient method.

    .. note::

        If you wish to use exact hessian, use the matrix-free :code:`tz.m.NewtonCGSteihaug`
        which only uses hessian-vector products. While passing ``tz.m.Newton`` to this
        is possible, it is usually less efficient.

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
        update_freq (int, optional): frequency of updating the hessian. Defaults to 1.
        reg (int, optional): regularization parameter for conjugate gradient. Defaults to 0.
        max_attempts (max_attempts, optional):
            maximum number of trust region size size reductions per step. A zero update vector is returned when
            this limit is exceeded. Defaults to 10.
        boundary_tol (float | None, optional):
            The trust region only increases when suggested step's norm is at least `(1-boundary_tol)*trust_region`.
            This prevents increasing trust region when solution is not on the boundary. Defaults to 1e-2.
        prefer_exact (bool, optional):
            when exact solution can be easily calculated without CG (e.g. hessian is stored as scaled identity),
            uses the exact solution. If False, always uses CG. Defaults to True.
        inner (Chainable | None, optional): preconditioning is applied to output of thise module. Defaults to None.

    Examples:
        Trust-SR1

        .. code-block:: python

            opt = tz.Optimizer(
                model.parameters(),
                tz.m.TrustCG(hess_module=tz.m.SR1(inverse=False)),
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
        boundary_tol: float | None = 1e-6, # tuned
        init: float = 1,
        max_attempts: int = 10,
        radius_strategy: _RadiusStrategy | _RADIUS_KEYS = 'default',
        reg: float = 0,
        maxiter: int | None = None,
        miniter: int = 1,
        cg_tol: float = 1e-8,
        prefer_exact: bool = True,
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):
        defaults = dict(reg=reg, prefer_exact=prefer_exact, cg_tol=cg_tol, maxiter=maxiter, miniter=miniter)
        super().__init__(
            defaults=defaults,
            hess_module=hess_module,
            eta=eta,
            nplus=nplus,
            nminus=nminus,
            rho_good=rho_good,
            rho_bad=rho_bad,
            boundary_tol=boundary_tol,
            init=init,
            max_attempts=max_attempts,
            radius_strategy=radius_strategy,
            update_freq=update_freq,
            inner=inner,

            radius_fn=torch.linalg.vector_norm,
        )

    def trust_solve(self, f, g, H, radius, params, closure, settings):
        if settings['prefer_exact'] and isinstance(H, linear_operator.ScaledIdentity):
            return H.solve_bounded(g, radius)

        x, _ = cg(H.matvec, g, trust_radius=radius, reg=settings['reg'], maxiter=settings["maxiter"], miniter=settings["miniter"], tol=settings["cg_tol"])
        return x