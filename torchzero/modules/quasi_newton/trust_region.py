from abc import ABC, abstractmethod
from typing import Any, Literal, cast, final

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

def tikhonov_(H: torch.Tensor, reg: float):
    if reg!=0: H.add_(torch.eye(H.size(-1), dtype=H.dtype, device=H.device).mul_(reg))
    return H

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
    def step(self, var):
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

                hessian_module.update_tensor(
                    tensor=_flatten_tensors(var.get_update()),
                    param = _flatten_tensors(params),
                    grad = _flatten_tensors(var.grad) if var.grad is not None else None,
                    loss = var.loss,
                    state=hessian_module.state[params[0]],
                    settings=hessian_module.defaults,
                )
                h_state = hessian_module.state[params[0]]
                if "B" in h_state:
                    P = h_state["B"]
                    is_inverse=False
                else:
                    P = h_state["H"]
                    is_inverse=True

            if self._update_freq != 0:
                self.global_state['B'] = P
                self.global_state['is_inverse'] = is_inverse

        if P is None:
            P = self.global_state['B']
            is_inverse = self.global_state['is_inverse']

        assert is_inverse is not None

        # -------------------------------- inner step -------------------------------- #
        update = var.get_update()
        if 'inner' in self.children:
            if g_list is None: g_list = var.grad
            update = apply_transform(self.children['inner'], update, params=params, grads=g_list, var=var)

        # ----------------------------------- apply ---------------------------------- #
        return self.trust_region_step(var=var, tensors=update, P=P, is_inverse=is_inverse)

class TrustNCG(TrustRegionBase):
    """Trust region via Steihaug-Toint Conjugate Gradient method. This is mainly useful for quasi-newton methods.
    If you don't use :code:`hess_module`, use the matrix-free :code:`tz.m.NewtonCGSteihaug` which only uses hessian-vector products.

    Args:
        hess_module (HessianUpdateStrategy | None, optional):
            Hessian update strategy, must be one of the :code:`HessianUpdateStrategy` modules. Make sure to set :code:`inverse=False`. If None, uses autograd to calculate the hessian. Defaults to None.
        init (float, optional): Initial trust region value. Defaults to 1.
        nplus (float, optional): increase factor on successful steps. Defaults to 1.5.
        nminus (float, optional): decrease factor on unsuccessful steps. Defaults to 0.75.
        update_freq (int, optional): frequency of updating the hessian. Defaults to 1.
        inner (Chainable | None, optional): preconditioning is applied to output of thise module. Defaults to None.

    Examples:
        Trust-Newton

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.ExactTrustRegion(),
            )

        Trust-SR1

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.ExactTrustRegion(hess_module=tz.m.SR1(inverse=False)),
            )
    """
    def __init__(
        self,
        hess_module: HessianUpdateStrategy | None = None,
        eta: float= 0.15,
        nplus: float = 2,
        nminus: float = 0.25,
        init: float = 1,
        update_freq: int = 1,
        reg: float = 0,
        inner: Chainable | None = None,
    ):
        defaults = dict(init=init, nplus=nplus, nminus=nminus, eta=eta, reg=reg)
        super().__init__(defaults, hess_module=hess_module, update_freq=update_freq, inner=inner)


    def trust_region_step(self, var, tensors, P, is_inverse):
        params = var.params
        settings = self.settings[params[0]]
        g = _flatten_tensors(tensors)

        trust_region = self.global_state.get('trust_region', settings['init'])
        if trust_region < 1e-8: trust_region = self.global_state['trust_region'] = settings['init']

        nplus = settings['nplus']
        nminus = settings['nminus']
        eta = settings['eta']
        reg = settings['reg']

        loss = var.loss
        closure = var.closure
        if closure is None: raise RuntimeError("Trust region requires closure")
        if loss is None: loss = closure(False)

        if is_inverse: P = torch.linalg.inv(P) #pylint:disable=not-callable # maybe there are better strats?
        update = steihaug_toint_cg(P, -g, trust_region, reg=reg)

        # evaluate actual loss reduction
        update_unflattned = vec_to_tensors(update, params)
        params = TensorList(params)
        params -= update_unflattned
        loss_star = closure(False)
        params += update_unflattned
        reduction = loss - loss_star

        # expected reduction is g.T @ p + 0.5 * p.T @ B @ p
        pred_reduction = - (g.dot(update) + 0.5 * update.dot(P @ update))
        rho = reduction / (pred_reduction.clip(min=1e-8))

        # failed step
        if rho < 0.25:
            self.global_state['trust_region'] = trust_region * nminus

        # very good step
        elif rho > 0.75:
            diff = trust_region - update.abs()
            if (diff.amin() / trust_region) > 1e-4: # hits boundary
                self.global_state['trust_region'] = trust_region * nplus

        # if the ratio is high enough then accept the proposed step
        if rho > eta:
            var.update = vec_to_tensors(update, params)

        else:
            var.update = params.zeros_like()
        return var

