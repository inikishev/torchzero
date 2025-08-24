# pylint:disable=not-callable
import torch

from ...core import Chainable, Module
from ...utils import TensorList, vec_to_tensors
from .trust_region import _RADIUS_KEYS, TrustRegionBase, _RadiusStrategy

class Dogleg(TrustRegionBase):
    """Dogleg trust region algorithm.


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
        max_attempts (max_attempts, optional):
            maximum number of trust region size size reductions per step. A zero update vector is returned when
            this limit is exceeded. Defaults to 10.
        inner (Chainable | None, optional): preconditioning is applied to output of thise module. Defaults to None.

    """
    def __init__(
        self,
        hess_module: Chainable,
        eta: float= 0.0,
        nplus: float = 2,
        nminus: float = 0.25,
        rho_good: float = 0.75,
        rho_bad: float = 0.25,
        boundary_tol: float | None = None,
        init: float = 1,
        max_attempts: int = 10,
        radius_strategy: _RadiusStrategy | _RADIUS_KEYS = 'default',
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):
        defaults = dict()
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
        if radius > 2: radius = self.global_state['radius'] = 2
        eps = torch.finfo(g.dtype).tiny * 2

        gHg = g.dot(H.matvec(g))
        if gHg <= eps:
            return (radius / torch.linalg.vector_norm(g)) * g # pylint:disable=not-callable

        p_cauchy = (g.dot(g) / gHg) * g
        p_newton = H.solve(g)

        a = p_newton - p_cauchy
        b = p_cauchy

        aa = a.dot(a)
        if aa < eps:
            return (radius / torch.linalg.vector_norm(g)) * g # pylint:disable=not-callable

        ab = a.dot(b)
        bb = b.dot(b)
        c = bb - radius**2
        discriminant = (2*ab)**2 - 4*aa*c
        beta = (-2*ab + torch.sqrt(discriminant.clip(min=0))) / (2 * aa)
        return p_cauchy + beta * (p_newton - p_cauchy)

