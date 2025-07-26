from typing import Literal

import torch

from ...core import Module, apply_transform, Chainable
from ...utils import NumberList, TensorList, as_tensorlist
from ...utils.derivatives import hvp, hvp_fd_central, hvp_fd_forward

class MatrixMomentum(Module):
    """Second order momentum method.

    Matrix momentum is useful for convex objectives, also for some reason it has very really good generalization on elastic net logistic regression.

    .. note::
        :code:`mu` needs to be tuned very carefully. It is supposed to be smaller than (1/largest eigenvalue), otherwise this will be very unstable.

    .. note::
        I have devised an adaptive version of this - :code:`tz.m.AdaptiveMatrixMomentum`, and it works well
        without having to tune :code:`mu`.

    .. note::
        In most cases MatrixMomentum should be the first module in the chain because it relies on autograd.

    .. note::
        This module requires the a closure passed to the optimizer step,
        as it needs to re-evaluate the loss and gradients for calculating HVPs.
        The closure must accept a ``backward`` argument (refer to documentation).

    Args:
        mu (float, optional): this has a similar role to (1 - beta) in normal momentum. Defaults to 0.1.
        beta (float, optional): decay for the buffer, this is not part of the original update rule. Defaults to 1.
        hvp_method (str, optional):
            Determines how Hessian-vector products are evaluated.

            - ``"autograd"``: Use PyTorch's autograd to calculate exact HVPs.
              This requires creating a graph for the gradient.
            - ``"forward"``: Use a forward finite difference formula to
              approximate the HVP. This requires one extra gradient evaluation.
            - ``"central"``: Use a central finite difference formula for a
              more accurate HVP approximation. This requires two extra
              gradient evaluations.
            Defaults to "autograd".
        h (float, optional): finite difference step size if hvp_method is set to finite difference. Defaults to 1e-3.
        hvp_tfm (Chainable | None, optional): optional module applied to hessian-vector products. Defaults to None.

    Reference:
        Orr, Genevieve, and Todd Leen. "Using curvature information for fast stochastic search." Advances in neural information processing systems 9 (1996).
    """

    def __init__(
        self,
        mu=0.1,
        beta: float = 1,
        hvp_method: Literal["autograd", "forward", "central"] = "autograd",
        h: float = 1e-3,
        hvp_tfm: Chainable | None = None,
    ):
        defaults = dict(mu=mu, beta=beta, hvp_method=hvp_method, h=h)
        super().__init__(defaults)

        if hvp_tfm is not None:
            self.set_child('hvp_tfm', hvp_tfm)

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('prev_update')

    @torch.no_grad
    def update(self, var):
        assert var.closure is not None
        prev_update = self.get_state(var.params, 'prev_update')
        hvp_method = self.settings[var.params[0]]['hvp_method']
        h = self.settings[var.params[0]]['h']

        Hvp, _ = self.Hvp(prev_update, at_x0=True, var=var, rgrad=None, hvp_method=hvp_method, h=h, normalize=True, retain_grad=False)
        Hvp = [t.detach() for t in Hvp]

        if 'hvp_tfm' in self.children:
            Hvp = TensorList(apply_transform(self.children['hvp_tfm'], Hvp, params=var.params, grads=var.grad, var=var))

        self.store(var.params, "Hvp", Hvp)


    @torch.no_grad
    def apply(self, var):
        update = TensorList(var.get_update())
        Hvp, prev_update = self.get_state(var.params, 'Hvp', 'prev_update', cls=TensorList)
        mu,beta = self.get_settings(var.params, 'mu','beta', cls=NumberList)

        update.add_(prev_update - Hvp*mu)
        prev_update.set_(update * beta)
        var.update = update
        return var


class AdaptiveMatrixMomentum(Module):
    """Second order momentum method.

    Matrix momentum is useful for convex objectives, also for some reason it has very good generalization on elastic net logistic regression.

    .. note::
        In most cases MatrixMomentum should be the first module in the chain because it relies on autograd.

    .. note::
        This module requires the a closure passed to the optimizer step,
        as it needs to re-evaluate the loss and gradients for calculating HVPs.
        The closure must accept a ``backward`` argument (refer to documentation).


    Args:
        mu_mul (float, optional): multiplier to the estimated mu. Defaults to 1.
        beta (float, optional): decay for the buffer, this is not part of the original update rule. Defaults to 1.
        hvp_method (str, optional):
            Determines how Hessian-vector products are evaluated.

            - ``"autograd"``: Use PyTorch's autograd to calculate exact HVPs.
              This requires creating a graph for the gradient.
            - ``"forward"``: Use a forward finite difference formula to
              approximate the HVP. This requires one extra gradient evaluation.
            - ``"central"``: Use a central finite difference formula for a
              more accurate HVP approximation. This requires two extra
              gradient evaluations.
            Defaults to "autograd".
        h (float, optional): finite difference step size if hvp_method is set to finite difference. Defaults to 1e-3.
        hvp_tfm (Chainable | None, optional): optional module applied to hessian-vector products. Defaults to None.

    Reference:
        Orr, Genevieve, and Todd Leen. "Using curvature information for fast stochastic search." Advances in neural information processing systems 9 (1996).
    """

    def __init__(
        self,
        mu_mul: float = 1,
        beta: float = 1,
        eps=1e-4,
        hvp_method: Literal["autograd", "forward", "central"] = "autograd",
        h: float = 1e-3,
        hvp_tfm: Chainable | None = None,
    ):
        defaults = dict(mu_mul=mu_mul, beta=beta, hvp_method=hvp_method, h=h, eps=eps)
        super().__init__(defaults)

        if hvp_tfm is not None:
            self.set_child('hvp_tfm', hvp_tfm)

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('prev_params', 'prev_grad')

    @torch.no_grad
    def update(self, var):
        assert var.closure is not None
        prev_update, prev_params, prev_grad = self.get_state(var.params, 'prev_update', 'prev_params', 'prev_grad', cls=TensorList)

        settings = self.settings[var.params[0]]
        hvp_method = settings['hvp_method']
        h = settings['h']
        eps = settings['eps']

        mu_mul = NumberList(self.settings[p]['mu_mul'] for p in var.params)

        Hvp, _ = self.Hvp(prev_update, at_x0=True, var=var, rgrad=None, hvp_method=hvp_method, h=h, normalize=True, retain_grad=False)
        Hvp = [t.detach() for t in Hvp]

        if 'hvp_tfm' in self.children:
            Hvp = TensorList(apply_transform(self.children['hvp_tfm'], Hvp, params=var.params, grads=var.grad, var=var))

        # adaptive part
        s_k = var.params - prev_params
        prev_params.copy_(var.params)

        if hvp_method != 'central': assert var.grad is not None
        grad = var.get_grad()
        y_k = grad - prev_grad
        prev_grad.copy_(grad)

        ada_mu = (s_k.global_vector_norm() / (y_k.global_vector_norm() + eps)) * mu_mul

        self.store(var.params, ['Hvp', 'ada_mu'], [Hvp, ada_mu])

    @torch.no_grad
    def apply(self, var):
        Hvp, ada_mu = self.get_state(var.params, 'Hvp', 'ada_mu')
        Hvp = as_tensorlist(Hvp)
        beta = NumberList(self.settings[p]['beta'] for p in var.params)
        update = TensorList(var.get_update())
        prev_update = TensorList(self.state[p]['prev_update'] for p in var.params)

        update.add_(prev_update - Hvp*ada_mu)
        prev_update.set_(update * beta)
        var.update = update
        return var

