from typing import Literal

import torch

from ...core import Module, apply_transform
from ...utils import NumberList, TensorList, as_tensorlist
from ...utils.derivatives import hvp, hvp_fd_central, hvp_fd_forward

class MatrixMomentum(Module):
    """
    May be useful for ill conditioned stochastic quadratic objectives but I need to test this.
    Evaluates hessian vector product on each step (via finite difference or autograd).

    `mu` is supposed to be smaller than (1/largest eigenvalue), otherwise this will be very unstable.

    Orr, Genevieve, and Todd Leen. "Using curvature information for fast stochastic search." Advances in neural information processing systems 9 (1996).
    """
    def __init__(self, mu=0.1, beta:float=1, hvp_method: Literal['autograd', 'forward', 'central'] = 'forward', h=1e-3, hvp_tfm=None):
        defaults = dict(mu=mu, beta=beta, hvp_method=hvp_method, h=h)
        super().__init__(defaults)

        if hvp_tfm is not None:
            self.set_child('hvp_tfm', hvp_tfm)

    @torch.no_grad
    def step(self, var):
        assert var.closure is not None
        prev_update = self.get_state('prev_update', params=var.params, cls=TensorList)
        hvp_method = self.settings[var.params[0]]['hvp_method']
        h = self.settings[var.params[0]]['h']

        mu,beta = self.get_settings('mu','beta', params=var.params, cls=NumberList)

        if hvp_method == 'autograd':
            with torch.enable_grad():
                grad = var.get_grad(create_graph=True)
                hvp_ = TensorList(hvp(var.params, grads=grad, vec=prev_update, allow_unused=True, retain_graph=False)).detach_()

        elif hvp_method == 'forward':
            var.get_grad()
            l, hvp_ = hvp_fd_forward(var.closure, var.params, vec=prev_update, g_0=var.grad, h=h, normalize=True)
            if var.loss_approx is None: var.loss_approx = l

        elif hvp_method == 'central':
            l, hvp_ = hvp_fd_central(var.closure, var.params, vec=prev_update, h=h, normalize=True)
            if var.loss_approx is None: var.loss_approx = l

        else:
            raise ValueError(hvp_method)

        if 'hvp_tfm' in self.children:
            hvp_ = TensorList(apply_transform(self.children['hvp_tfm'], hvp_, params=var.params, grads=var.grad, var=var))

        update = TensorList(var.get_update())

        hvp_ = as_tensorlist(hvp_)
        update.add_(prev_update - hvp_*mu)
        prev_update.set_(update * beta)
        var.update = update
        return var


class AdaptiveMatrixMomentum(Module):
    """
    Mu here is estimated as ||s_k||/||y_k||.
    """
    def __init__(self, mu_mul:float=1, beta:float=1, eps=1e-4, hvp_method: Literal['autograd', 'forward', 'central'] = 'forward', h=1e-3, hvp_tfm=None):
        defaults = dict(mu_mul=mu_mul, beta=beta, hvp_method=hvp_method, h=h, eps=eps)
        super().__init__(defaults)

        if hvp_tfm is not None:
            self.set_child('hvp_tfm', hvp_tfm)

    @torch.no_grad
    def step(self, var):
        assert var.closure is not None
        prev_update, prev_params, prev_grad = self.get_state('prev_update', 'prev_params', 'prev_grad', params=var.params, cls=TensorList)

        settings = self.settings[var.params[0]]
        hvp_method = settings['hvp_method']
        h = settings['h']
        eps = settings['eps']

        mu_mul, beta = self.get_settings('mu_mul','beta', params=var.params, cls=NumberList)

        if hvp_method == 'autograd':
            with torch.enable_grad():
                grad = var.get_grad(create_graph=True)
                hvp_ = TensorList(hvp(var.params, grads=grad, vec=prev_update, allow_unused=True, retain_graph=False)).detach_()

        elif hvp_method == 'forward':
            var.get_grad()
            l, hvp_ = hvp_fd_forward(var.closure, var.params, vec=prev_update, g_0=var.grad, h=h, normalize=True)
            if var.loss_approx is None: var.loss_approx = l

        elif hvp_method == 'central':
            l, hvp_ = hvp_fd_central(var.closure, var.params, vec=prev_update, h=h, normalize=True)
            if var.loss_approx is None: var.loss_approx = l

        else:
            raise ValueError(hvp_method)

        if 'hvp_tfm' in self.children:
            hvp_ = TensorList(apply_transform(self.children['hvp_tfm'], hvp_, params=var.params, grads=var.grad, var=var))

        # adaptive part
        update = TensorList(var.get_update())

        s_k = var.params - prev_params
        prev_params.copy_(var.params)

        assert var.grad is not None
        y_k = var.grad - prev_grad
        prev_grad.copy_(var.grad)

        ada_mu = (s_k.global_vector_norm() / (y_k.global_vector_norm() + eps)) * mu_mul

        # matrix momentum uppdate
        hvp_ = as_tensorlist(hvp_)
        update.add_(prev_update - hvp_*ada_mu)
        prev_update.set_(update * beta)
        var.update = update
        return var

