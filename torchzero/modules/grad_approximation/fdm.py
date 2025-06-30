from collections.abc import Callable
from typing import Any

import torch

from ...utils import TensorList
from .grad_approximator import GradApproximator, GradTarget, _FD_Formula


def _forward2(closure: Callable[..., float], param:torch.Tensor, idx: int, h, v_0: float | None):
    if v_0 is None: v_0 = closure(False)
    assert param.ndim == 1
    param[idx] += h
    v_plus = closure(False)
    param[idx] -= h
    return v_0, v_0, (v_plus - v_0) / h # (loss, loss_approx, grad)

def _backward2(closure: Callable[..., float], param:torch.Tensor, idx: int, h, v_0: float | None):
    if v_0 is None: v_0 = closure(False)
    assert param.ndim == 1
    param[idx] -= h
    v_minus = closure(False)
    param[idx] += h
    return v_0, v_0, (v_0 - v_minus) / h

def _central2(closure: Callable[..., float], param:torch.Tensor, idx: int, h, v_0: Any):
    assert param.ndim == 1
    param[idx] += h
    v_plus = closure(False)

    param[idx] -= h * 2
    v_minus = closure(False)

    param[idx] += h
    return v_0, v_plus, (v_plus - v_minus) / (2 * h)

def _forward3(closure: Callable[..., float], param:torch.Tensor, idx: int, h, v_0: float | None):
    if v_0 is None: v_0 = closure(False)
    assert param.ndim == 1
    param[idx] += h
    v_plus1 = closure(False)

    param[idx] += h
    v_plus2 = closure(False)

    param[idx] -= 2 * h
    return v_0, v_0, (-3*v_0 + 4*v_plus1 - v_plus2) / (2 * h)

def _backward3(closure: Callable[..., float], param:torch.Tensor, idx: int, h, v_0: float | None):
    if v_0 is None: v_0 = closure(False)
    assert param.ndim == 1
    param[idx] -= h
    v_minus1 = closure(False)

    param[idx] -= h
    v_minus2 = closure(False)

    param[idx] += 2 * h
    return v_0, v_0, (v_minus2 - 4*v_minus1 + 3*v_0) / (2 * h)

def _central4(closure: Callable[..., float], param:torch.Tensor, idx: int, h, v_0: Any):
    assert param.ndim == 1

    param[idx] += h
    v_plus1 = closure(False)

    param[idx] += h
    v_plus2 = closure(False)

    param[idx] -= 3 * h
    v_minus1 = closure(False)

    param[idx] -= h
    v_minus2 = closure(False)

    param[idx] += 2 * h
    return v_0, v_plus1, (v_minus2 - 8*v_minus1 + 8*v_plus1 - v_plus2) / (12 * h)

_FD_FUNCS = {
    "forward": _forward2,
    "forward2": _forward2,
    "backward": _backward2,
    "backward2": _backward2,
    "central": _central2,
    "central2": _central2,
    "central3": _central2, # they are the same
    "forward3": _forward3,
    "backward3": _backward3,
    "central4": _central4,
}


class FDM(GradApproximator):
    """Approximate gradients via finite difference method.

    .. note::
        This module is a gradient approximator. It modifies the closure to evaluate the estimated gradients,
        and further closure-based modules will use the modified closure. All modules after this will use estimated gradients.

    Args:
        h (float, optional): magnitude of parameter perturbation. Defaults to 1e-3.
        formula (_FD_Formula, optional): finite difference formula. Defaults to 'central2'.
        target (GradTarget, optional): what to set on var. Defaults to 'closure'.

    Examples:
        plain FDM:

        .. code-block:: python

            fdm = tz.Modular(model.parameters(), tz.m.FDM(), tz.m.LR(1e-2))

        Any gradient-based method can use FDM-estimated gradients seamlessly.

        .. code-block:: python

            fdm_ncg = tz.Modular(
                model.parameters(),
                tz.m.FDM(),
                # set hvp_method to "forward" so that it
                # uses gradient difference instead of autograd
                tz.m.NewtonCG(hvp_method="forward"),
                tz.m.Backtracking()
            )
    """
    def __init__(self, h: float=1e-3, formula: _FD_Formula = 'central', target: GradTarget = 'closure'):
        defaults = dict(h=h, formula=formula)
        super().__init__(defaults, target=target)

    @torch.no_grad
    def approximate(self, closure, params, loss):
        grads = []
        loss_approx = None

        for p in params:
            g = torch.zeros_like(p)
            grads.append(g)

            settings = self.settings[p]
            h = settings['h']
            fd_fn = _FD_FUNCS[settings['formula']]

            p_flat = p.view(-1); g_flat = g.view(-1)
            for i in range(len(p_flat)):
                loss, loss_approx, d = fd_fn(closure=closure, param=p_flat, idx=i, h=h, v_0=loss)
                g_flat[i] = d

        return grads, loss, loss_approx