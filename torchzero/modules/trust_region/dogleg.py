# pylint:disable=not-callable
from collections.abc import Callable

import torch

from ...core import Chainable, Module
from ...utils import TensorList, vec_to_tensors
from ...utils.linalg import linear_operator

from .trust_region import TrustRegionBase, _update_tr_radius

def _flatten_tensors(tensors: list[torch.Tensor]):
    return torch.cat([t.ravel() for t in tensors])

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
        hess_module: Module,
        eta: float= 0.0,
        nplus: float = 2,
        nminus: float = 0.25,
        rho_good: float = 0.75,
        rho_bad: float = 0.25,
        init: float = 1,
        update_freq: int = 1,
        max_attempts: int = 10,
        boundary_tol=None,
        inner: Chainable | None = None,
    ):
        defaults = dict(init=init, nplus=nplus, nminus=nminus, eta=eta, max_attempts=max_attempts, rho_bad=rho_bad, rho_good=rho_good, boundary_tol=boundary_tol)
        super().__init__(hess_module=hess_module, defaults=defaults, update_freq=update_freq, inner=inner)

    @torch.no_grad
    def trust_region_apply(self, var, tensors, H):
        params = TensorList(var.params)
        settings = self.settings[params[0]]

        assert H is not None
        g = _flatten_tensors(tensors)

        max_attempts = settings['max_attempts']

        loss = var.loss
        closure = var.closure
        if closure is None: raise RuntimeError("Trust region requires closure")
        if loss is None: loss = var.get_loss(False)

        success = False
        d = None
        while not success:
            max_attempts -= 1
            if max_attempts < 0: break

            trust_region = min(self.global_state.get('trust_region', settings['init']), 2)

            if trust_region < 1e-12:
                trust_region = self.global_state['trust_region'] = settings['init']

            gHg = g.dot(H.matvec(g))
            if gHg <= 1e-12:
                d = (trust_region / torch.linalg.vector_norm(g)) * g # pylint:disable=not-callable
            else:
                p_cauchy = (g.dot(g) / gHg) * g
                p_newton = H.solve(g)

                a = p_newton - p_cauchy
                b = p_cauchy

                aa = a.dot(a)
                if aa < 1e-12:
                    d = (trust_region / torch.linalg.vector_norm(g)) * g # pylint:disable=not-callable
                else:
                    ab = a.dot(b)
                    bb = b.dot(b)
                    c = bb - trust_region**2
                    discriminant = (2*ab)**2 - 4*aa*c
                    beta = (-2*ab + torch.sqrt(discriminant.clip(min=0))) / (2 * aa)
                    d = p_cauchy + beta * (p_newton - p_cauchy)

            self.global_state['trust_region'], success = _update_tr_radius(
                params=params, closure=closure, d=d, f=loss, g=g, H=H,
                trust_region=trust_region, settings=settings,
            )

        assert d is not None
        if success: var.update = vec_to_tensors(d, params)
        else: var.update = params.zeros_like()

        return var

