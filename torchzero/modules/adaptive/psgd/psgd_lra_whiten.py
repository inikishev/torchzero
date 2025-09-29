# pylint:disable=not-callable
"""all functions are from https://github.com/lixilinx/psgd_torch/blob/master/psgd.py"""
import math
import warnings

import torch

from ....core import Chainable, TensorTransform
from ._psgd_utils import _initialize_lra_state_
from .psgd import lift2single, precond_grad_lra, update_precond_lra_whiten

# matches
class LRAWhiten(TensorTransform):
    def __init__(
        self,
        rank: int = 10,
        init_scale: float | None = None,
        lr_preconditioner=0.1,
        betaL=0.9,
        damping=1e-9,
        momentum=0.0,
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