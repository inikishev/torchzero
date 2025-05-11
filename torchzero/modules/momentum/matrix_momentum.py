from typing import Literal

import torch

from ...core import Module, apply
from ...utils import NumberList, TensorList, as_tensorlist
from ...utils.derivatives import hvp, hvp_fd_central, hvp_fd_forward

class MatrixMomentum(Module):
    """
    May be useful for ill conditioned stochastic quadratic objectives but I need to test this.
    Evaluates hessian vector product on each step (via finite difference or autograd).

    `mu` is supposed to be smaller than (1/largest eigenvalue), otherwise this will be very unstable.

    Orr, Genevieve, and Todd Leen. "Using curvature information for fast stochastic search." Advances in neural information processing systems 9 (1996).
    """
    def __init__(self, mu=0.1, beta:float=1, hvp_mode: Literal['autograd', 'forward', 'central'] = 'forward', h=1e-3, hvp_tfm=None):
        defaults = dict(mu=mu, beta=beta, hvp_mode=hvp_mode, h=h)
        super().__init__(defaults)

        if hvp_tfm is not None:
            self.set_child('hvp_tfm', hvp_tfm)

    @torch.no_grad
    def step(self, vars):
        assert vars.closure is not None
        prev_update = self.get_state('prev_update', params=vars.params, cls=TensorList)
        hvp_mode = self.settings[vars.params[0]]['hvp_mode']
        h = self.settings[vars.params[0]]['h']

        mu,beta = self.get_settings('mu','beta', params=vars.params, cls=NumberList)

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
            l, hvp_ = hvp_fd_forward(vars.closure, vars.params, vec=prev_update, g_0=vars.grad, h=h, normalize=True)
            if vars.loss_approx is None: vars.loss_approx = l

        elif hvp_mode == 'central':
            l, hvp_ = hvp_fd_central(vars.closure, vars.params, vec=prev_update, h=h, normalize=True)
            if vars.loss_approx is None: vars.loss_approx = l

        else:
            raise ValueError(hvp_mode)

        if 'hvp_tfm' in self.children:
            hvp_ = TensorList(apply(self.children['hvp_tfm'], hvp_, params=vars.params, grads=vars.grad, vars=vars))

        update = TensorList(vars.get_update())

        hvp_ = as_tensorlist(hvp_)
        update.add_(prev_update - hvp_*mu)
        prev_update.set_(update * beta)
        vars.update = update
        return vars


class AdaptiveMatrixMomentum(Module):
    """
    Mu here is estimated as ||s_k||/||y_k||.
    """
    def __init__(self, mu_mul:float=1, beta:float=0.99, eps=1e-4, hvp_mode: Literal['autograd', 'forward', 'central'] = 'forward', h=1e-3, hvp_tfm=None):
        defaults = dict(mu_mul=mu_mul, beta=beta, hvp_mode=hvp_mode, h=h, eps=eps)
        super().__init__(defaults)

        if hvp_tfm is not None:
            self.set_child('hvp_tfm', hvp_tfm)

    @torch.no_grad
    def step(self, vars):
        assert vars.closure is not None
        prev_update, prev_params, prev_grad = self.get_state('prev_update', 'prev_params', 'prev_grad', params=vars.params, cls=TensorList)

        settings = self.settings[vars.params[0]]
        hvp_mode = settings['hvp_mode']
        h = settings['h']
        eps = settings['eps']

        mu_mul, beta = self.get_settings('mu_mul','beta', params=vars.params, cls=NumberList)

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
            l, hvp_ = hvp_fd_forward(vars.closure, vars.params, vec=prev_update, g_0=vars.grad, h=h, normalize=True)
            if vars.loss_approx is None: vars.loss_approx = l

        elif hvp_mode == 'central':
            l, hvp_ = hvp_fd_central(vars.closure, vars.params, vec=prev_update, h=h, normalize=True)
            if vars.loss_approx is None: vars.loss_approx = l

        else:
            raise ValueError(hvp_mode)

        if 'hvp_tfm' in self.children:
            hvp_ = TensorList(apply(self.children['hvp_tfm'], hvp_, params=vars.params, grads=vars.grad, vars=vars))

        # adaptive part
        update = TensorList(vars.get_update())

        s_k = vars.params - prev_params
        prev_params.copy_(vars.params)

        assert vars.grad is not None
        y_k = vars.grad - prev_grad
        prev_grad.copy_(vars.grad)

        ada_mu = (s_k.global_vector_norm() / (y_k.global_vector_norm() + eps)) * mu_mul

        # matrix momentum uppdate
        hvp_ = as_tensorlist(hvp_)
        update.add_(prev_update - hvp_*ada_mu)
        prev_update.set_(update * beta)
        vars.update = update
        return vars

