import torch

from ...core import Chainable, Module
from ...utils import TensorList, vec_to_tensors
from ...utils.linalg import cg
from .trust_region import TrustRegionBase, _update_tr_radius


def _flatten_tensors(tensors: list[torch.Tensor]):
    return torch.cat([t.ravel() for t in tensors])

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
        fallback (bool, optional):
            if ``True``, when ``hess_module`` maintains hessian inverse which can't be inverted efficiently, it will
            be inverted anyway. When ``False`` (default), a ``RuntimeError`` will be raised instead.
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
        rho_good: float = 0.75,
        rho_bad: float = 0.25,
        init: float = 1,
        update_freq: int = 1,
        reg: float = 0,
        max_attempts: int = 10,
        boundary_tol: float | None = 1e-1,
        fallback: bool = False,
        inner: Chainable | None = None,
    ):
        defaults = dict(init=init, nplus=nplus, nminus=nminus, eta=eta, reg=reg, max_attempts=max_attempts,boundary_tol=boundary_tol, rho_bad=rho_bad, rho_good=rho_good)
        super().__init__(hess_module=hess_module, requires="B", defaults=defaults, update_freq=update_freq, inner=inner, fallback=fallback)

    @torch.no_grad
    def trust_region_apply(self, var, tensors, B, H):
        assert B is not None

        params = TensorList(var.params)
        settings = self.settings[params[0]]
        g = _flatten_tensors(tensors)

        reg = settings['reg']
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

