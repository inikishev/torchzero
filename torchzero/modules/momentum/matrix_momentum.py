from typing import Literal

import torch

from ...core import Module, apply
from ...utils import NumberList, TensorList, set_storage_
from ...utils.derivatives import hvp, hvp_fd_central, hvp_fd_forward

class MatrixMomentum(Module):
    """
    `u` is supposed to be smaller than the (1/largest eigenvalue), otherwise this will be
    very unstable.


    Orr, Genevieve, and Todd Leen. "Using curvature information for fast stochastic search." Advances in neural information processing systems 9 (1996)."""
    def __init__(self, u=0.1, decay:float=0.99, adaptive=False, hvp_mode: Literal['autograd', 'forward', 'central'] = 'forward', h=1e-3, hvp_tfm=None):
        defaults = dict(u=u, decay=decay, hvp_mode=hvp_mode, adaptive=adaptive, h=h)
        super().__init__(defaults)

        if hvp_tfm is not None:
            self.set_child('hvp_tfm', hvp_tfm)

    @torch.no_grad
    def step(self, vars):
        assert vars.closure is not None
        prev_update = self.get_state('prev_update', params=vars.params, cls=TensorList)
        hvp_mode = self.settings[vars.params[0]]['hvp_mode']
        h = self.settings[vars.params[0]]['h']
        adaptive = self.settings[vars.params[0]]['adaptive']

        u,decay = self.get_settings('u','decay', params=vars.params, cls=NumberList)

        if hvp_mode == 'autograd':
            with torch.enable_grad():
                vars.zero_grad()
                vars.loss = vars.loss_approx = vars.closure(False)
                assert vars.loss is not None
                vars.loss.backward(create_graph=True)
                vars.grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in vars.params]

                hvp_ = TensorList(hvp(vars.params, grads=vars.grad, vec=prev_update, allow_unused=True, retain_graph=False)).detach_()

        elif hvp_mode == 'forward':
            vars.get_grad()
            hvp_ = TensorList(hvp_fd_forward(vars.closure, vars.params, vec=prev_update, g_0=vars.grad, h=h, normalize=True))

        elif hvp_mode == 'central':
            hvp_ = TensorList(hvp_fd_central(vars.closure, vars.params, vec=prev_update, h=h, normalize=True))

        else:
            raise ValueError(hvp_mode)

        if 'hvp_tfm' in self.children:
            hvp_ = TensorList(apply(self.children['hvp_tfm'], hvp_, params=vars.params, grad=vars.grad, vars=vars))

        update = TensorList(vars.get_update())

        if adaptive: hvp_.graft_(prev_update)
        update.add_(prev_update - hvp_*u)
        prev_update.set_(update * decay)
        vars.update = update
        return vars

