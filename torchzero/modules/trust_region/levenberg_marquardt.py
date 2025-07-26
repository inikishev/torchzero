# pylint:disable=not-callable
from collections.abc import Callable

import torch

from ...core import Chainable, Module
from ...utils import TensorList, vec_to_tensors
from ...utils.linalg import linear_operator

from .trust_region import TrustRegionBase, _update_tr_radius

def _flatten_tensors(tensors: list[torch.Tensor]):
    return torch.cat([t.ravel() for t in tensors])

class LevenbergMarquardt(TrustRegionBase):
    """Levenberg-Marquardt trust region algorithm.


    Args:
        hess_module (Module | None, optional):
            A module that maintains a hessian approximation (not hessian inverse!).
            This includes all full-matrix quasi-newton methods, ``tz.m.Newton`` and ``tz.m.GaussNewton``.
            When using quasi-newton methods, set `inverse=False` when constructing them.
        y (float, optional):
            when ``y=0``, identity matrix is added to hessian, when ``y=1``, diagonal of the hessian approximation
            is added. Values between interpolate. This should only be used with Gauss-Newton. Defaults to 0.
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
        fallback (bool, optional):
            if ``True``, when ``hess_module`` maintains hessian inverse which can't be inverted efficiently, it will
            be inverted anyway. When ``False`` (default), a ``RuntimeError`` will be raised instead.
        inner (Chainable | None, optional): preconditioning is applied to output of thise module. Defaults to None.

    Examples:
        Gauss-Newton with Levenberg-Marquardt trust-region

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.LevenbergMarquardt(tz.m.GaussNewton()),
            )

        LM-SR1

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.LevenbergMarquardt(tz.m.SR1(inverse=False)),
            )

        First order trust region (hessian is assumed to be identity)

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.LevenbergMarquardt(tz.m.Identity()),
            )

    """
    def __init__(
        self,
        hess_module: Module,
        y: float = 0,
        eta: float= 0.0,
        nplus: float = 3.5,
        nminus: float = 0.25,
        rho_good: float = 0.99,
        rho_bad: float = 1e-4,
        init: float = 1,
        update_freq: int = 1,
        max_attempts: int = 10,
        fallback: bool = False,
        inner: Chainable | None = None,
    ):
        defaults = dict(y=y, init=init, nplus=nplus, nminus=nminus, eta=eta, max_attempts=max_attempts, rho_bad=rho_bad, rho_good=rho_good, fallback=fallback)
        super().__init__(hess_module=hess_module, defaults=defaults, update_freq=update_freq, inner=inner)

    @torch.no_grad
    def trust_region_apply(self, var, tensors, H):
        params = TensorList(var.params)
        settings = self.settings[params[0]]

        assert H is not None
        if isinstance(H, linear_operator.DenseInverse):
            if settings['fallback']:
                H = H.to_dense()
            else:
                raise RuntimeError(
                    f"{self.children['hess_module']} maintains a hessian inverse. "
                    "LevenbergMarquardt requires the hessian, not the inverse. "
                    "If that module is a quasi-newton module, pass `inverse=False` on initialization. "
                    "Or pass `fallback=True` to LevenbergMarquardt to allow inverting the hessian inverse, "
                    "however that can be inefficient and unstable."
                )

        g = _flatten_tensors(tensors)

        max_attempts = settings['max_attempts']
        y = settings['y']

        loss = var.loss
        closure = var.closure
        if closure is None: raise RuntimeError("Trust region requires closure")
        if loss is None: loss = var.get_loss(False)

        success = False
        d = None
        while not success:
            max_attempts -= 1
            if max_attempts < 0: break

            trust_region = self.global_state.get('trust_region', settings['init'])

            if trust_region < 1e-12 or trust_region > 1e12:
                trust_region = self.global_state['trust_region'] = settings['init']

            reg = 1/trust_region
            if y == 0:
                d = H.add_diagonal(reg).solve(g)
            else:
                diag = H.diagonal()
                diag = torch.where(diag < 1e-10, 1, diag)
                if y != 1: diag = (diag*y) + (1-y)
                d = H.add_diagonal(diag*reg).solve(g)

            self.global_state['trust_region'], success = _update_tr_radius(
                params=params, closure=closure, d=d, f=loss, g=g, H=H,
                trust_region=trust_region, settings = settings, boundary_fn=None,
            )

        assert d is not None
        if success: var.update = vec_to_tensors(d, params)
        else: var.update = params.zeros_like()

        return var

