# pylint:disable=not-callable
"""all functions are from https://github.com/lixilinx/psgd_torch/blob/master/psgd.py"""
import math
import warnings

import torch

from ....core import Chainable, TensorTransform
from ._psgd_utils import _initialize_lra_state_
from .psgd import lift2single, precond_grad_lra, update_precond_lra_whiten

# matches
class PSGDLRAWhiten(TensorTransform):
    """Low rank whitening preconditioner from Preconditioned Stochastic Gradient Descent (see https://github.com/lixilinx/psgd_torch)

    Args:
        rank (int, optional):
            Preconditioner has a diagonal part and a low rank part, whose rank is decided by this setting. Defaults to 10.
        init_scale (float | None, optional):
            initial scale of the preconditioner. If None, determined based on a heuristic. Defaults to None.
        lr_preconditioner (float, optional): learning rate of the preconditioner. Defaults to 0.1.
        betaL (float, optional): EMA factor for the L-smoothness constant wrt Q. Defaults to 0.9.
        damping (float, optional):
            adds small noise to hessian-vector product when updating the preconditioner. Defaults to 1e-9.
        grad_clip_max_norm (float, optional): clips norm of the update. Defaults to float("inf").
        update_probability (float, optional): probability of updating preconditioner on each step. Defaults to 1.0.
        concat_params (bool, optional):
            if True, treats all parameters as concatenated to a single vector.
            If False, each parameter is preconditioned separately. Defaults to True.
        inner (Chainable | None, optional): preconditioning will be applied to output of this module. Defaults to None.

    ###Examples:

    Pure PSGD LRA:
    ```py
    optimizer = tz.Optimizer(
        model.parameters(),
        tz.m.LRAWhiten(),
        tz.m.LR(1e-3),
    )
    ```

    Momentum into preconditioner (whitens momentum):
    ```py
    optimizer = tz.Optimizer(
        model.parameters(),
        tz.m.EMA(0.9),
        tz.m.LRAWhiten(),
        tz.m.LR(1e-3),
    )
    ```

    Updating the preconditioner from gradients and applying it to momentum:
    ```py
    optimizer = tz.Optimizer(
        model.parameters(),
        tz.m.LRAWhiten(inner=tz.m.EMA(0.9)),
        tz.m.LR(1e-3),
    )
    ```

    """
    def __init__(
        self,
        rank: int = 10,
        init_scale: float | None = None,
        lr_preconditioner=0.1,
        betaL=0.9,
        damping=1e-9,
        grad_clip_max_amp=float("inf"),
        update_probability=1.0,

        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults["inner"], defaults["self"]
        super().__init__(defaults, concat_params=concat_params, inner=inner)

    @torch.no_grad
    def single_tensor_initialize(self, tensor, param, grad, loss, state, setting):
        _initialize_lra_state_(tensor, state, setting)

    @torch.no_grad
    def single_tensor_update(self, tensor, param, grad, loss, state, setting):

        g = tensor.ravel().unsqueeze(1) # column vector

        UVd = state["UVd"]
        if UVd[2] is None: # initialize d on the fly
            UVd[2] = (torch.mean(g**4) + setting["damping"]**4)**(-1/8) * torch.ones_like(g)

        if torch.rand([]) < setting["update_probability"]:  # update preconditioner
            update_precond_lra_whiten(
                UVd=UVd,
                Luvd=state["Luvd"],
                g=g,
                lr=setting["lr_preconditioner"],
                betaL=setting["betaL"],
                damping=setting["damping"],
            )

    @torch.no_grad
    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):

        g = tensor.ravel().unsqueeze(1)
        pre_grad = precond_grad_lra(UVd=state["UVd"], g=g)

        # norm clipping
        grad_clip_max_amp = setting["grad_clip_max_amp"]
        if grad_clip_max_amp < float("inf"): # clip preconditioned gradient
            amp = torch.sqrt(torch.mean(pre_grad * pre_grad))
            if amp > grad_clip_max_amp:
                pre_grad *= grad_clip_max_amp/amp

        return pre_grad.view_as(tensor)