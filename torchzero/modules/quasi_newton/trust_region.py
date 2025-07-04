import warnings
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

import numpy as np
import torch
from scipy.optimize import lsq_linear

from ...core import Chainable, Module, TensorwiseTransform, Transform, apply_transform
from ...utils import TensorList, vec_to_tensors, vec_to_tensors_
from ...utils.derivatives import (
    hessian_list_to_mat,
    hessian_mat,
    hvp,
    hvp_fd_central,
    hvp_fd_forward,
    jacobian_and_hessian_wrt,
)
from .quasi_newton import HessianUpdateStrategy


def trust_lstsq(H: torch.Tensor, g: torch.Tensor, trust_region: float):
    res = lsq_linear(H.numpy(force=True).astype(np.float64), g.numpy(force=True).astype(np.float64), bounds=(-trust_region, trust_region))
    x = torch.from_numpy(res.x).to(H)
    return x, res.cost

def tikhonov_(H: torch.Tensor, reg: float):
    if reg!=0: H.add_(torch.eye(H.size(-1), dtype=H.dtype, device=H.device).mul_(reg))
    return H

def _flatten_tensors(tensors: list[torch.Tensor]):
    return torch.cat([t.ravel() for t in tensors])

class ExactTrustRegion(Module):
    """Exact trust region.

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
        init: float = 1,
        nplus: float = 1.5,
        nminus: float = 0.75,
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):

        defaults = dict(init=init, nplus=nplus, nminus=nminus, update_freq=update_freq)
        super().__init__(defaults)

        if hess_module is not None:
            self.set_child('hess_module', hess_module)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def step(self, var):
        # ---------------------------------- update ---------------------------------- #
        closure = var.closure
        if closure is None: raise RuntimeError("Trust region requires closure")
        params = var.params
        settings = self.settings[params[0]]
        update_freq = settings['update_freq']

        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        B = None
        g_list = var.grad
        loss = var.loss
        if step % update_freq == 0:

            if 'hess_module' not in self.children:
                params=var.params
                closure=var.closure
                if closure is None: raise ValueError('Closure is required if hessian_module is set to "newton"')
                with torch.enable_grad():
                    loss = var.loss = var.loss_approx = closure(False)
                    g_list, H_list = jacobian_and_hessian_wrt([loss], params, batched=True)
                    g_list = [t[0] for t in g_list] # remove leading dim from loss
                    var.grad = g_list
                    B = hessian_list_to_mat(H_list)


            else:
                hessian_module = cast(HessianUpdateStrategy | None, self.children.get('hess_module', None))

                if hessian_module is not None:
                    hessian_module.update_tensor(
                        tensor=_flatten_tensors(var.get_update()),
                        param = _flatten_tensors(params),
                        grad = _flatten_tensors(var.grad) if var.grad is not None else None,
                        loss = var.loss,
                        state=hessian_module.state[params[0]],
                        settings=hessian_module.defaults,
                    )
                    B = hessian_module.state[params[0]]['B']

            if update_freq != 0: self.global_state['B'] = B

        if B is None:
            B = self.global_state['B']

        # -------------------------------- inner step -------------------------------- #
        update = var.get_update()
        if 'inner' in self.children:
            if g_list is None: g_list = var.grad
            update = apply_transform(self.children['inner'], update, params=params, grads=g_list, var=var)

        g = _flatten_tensors(update)

        # ----------------------------------- apply ---------------------------------- #
        trust_region = self.global_state.get('trust_region', settings['init'])
        nplus = settings['nplus']
        nminus = settings['nminus']

        if loss is None: loss = closure(False)
        assert B is not None
        update, cost = trust_lstsq(B, g, trust_region)

        update_unflattned = vec_to_tensors(update, params)
        params = TensorList(params)
        params -= update_unflattned
        loss_star = closure(False)
        params += update_unflattned
        reduction = loss - loss_star

        # failed step
        if reduction <= 0:
            self.global_state['trust_region'] = trust_region * nminus
            update.zero_()

        # good step
        else:
            self.global_state['trust_region'] = trust_region * nplus

        var.update = vec_to_tensors(update, params)
        return var
