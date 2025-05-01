from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import torch

from ...utils import Distributions, NumberList, TensorList, generic_eq
from ...utils.derivatives import jvp, jvp_fd_central, jvp_fd_forward
from .grad_approximator import GradApproximator, GradTarget
from .rfdm import RandomizedFDM


class ForwardGradient(RandomizedFDM):
    PRE_MULTIPLY_BY_H = False
    def __init__(
        self,
        n_samples: int = 1,
        distribution: Distributions = "gaussian",
        beta: float = 0,
        jvp_method: Literal['autograd', 'forward', 'central'] = 'autograd',
        h: float = 1e-3,
        target: GradTarget = "closure",
    ):
        super().__init__(h=h, n_samples=n_samples, distribution=distribution, beta=beta, target=target)
        self.defaults['jvp_method'] = jvp_method

    @torch.no_grad
    def approximate(self, closure, params, loss, vars):
        params = TensorList(params)
        loss_approx = None

        settings = self.settings[params[0]]
        n_samples = settings['n_samples']
        jvp_method = settings['jvp_method']
        h = settings['h']
        perturbations = self.global_state['perturbations']

        grad = None
        for i in range(n_samples):
            prt = perturbations[i]
            if jvp_method == 'autograd':
                with torch.enable_grad():
                    loss, d = jvp(partial(closure, False), params=params, tangent=prt)

            elif jvp_method == 'forward':
                loss, d = jvp_fd_forward(partial(closure, False), params=params, tangent=prt, v_0=loss, normalize=True, h=h)

            elif jvp_method == 'central':
                loss_approx, d = jvp_fd_central(partial(closure, False), params=params, tangent=prt, normalize=True, h=h)

            else: raise ValueError(jvp_method)

            if grad is None: grad = prt * d
            else: grad += prt * d

        assert grad is not None
        return grad, loss, loss_approx

