from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import torch

from ...utils import Distributions, NumberList, TensorList
from ...utils.derivatives import jvp, jvp_fd_central, jvp_fd_forward
from .grad_approximator import GradApproximator, GradTarget
from .rfdm import RandomizedFDM


class ForwardGradient(RandomizedFDM):
    """Forward gradient method.

    This method samples one or more directional derivatives evaluated via autograd jacobian-vector products. This is very similar to randomized finite difference.

    Note:
        This module is a gradient approximator. It modifies the closure to evaluate the estimated gradients,
        and further closure-based modules will use the modified closure. All modules after this will use estimated gradients.


    Args:
        n_samples (int, optional): number of random gradient samples. Defaults to 1.
        distribution (Distributions, optional): distribution for random gradient samples. Defaults to "gaussian".
        pre_generate (bool, optional):
            whether to pre-generate gradient samples before each step. If samples are not pre-generated, whenever a method performs multiple closure evaluations, the gradient will be evaluated in different directions each time. Defaults to True.
        jvp_method (str, optional):
            how to calculate jacobian vector product, note that with `forward` and 'central' this is equivalent to randomized finite difference. Defaults to 'autograd'.
        h (float, optional): finite difference step size of jvp_method is set to `forward` or `central`. Defaults to 1e-3.
        target (GradTarget, optional): what to set on var. Defaults to "closure".

    References:
        Baydin, A. G., Pearlmutter, B. A., Syme, D., Wood, F., & Torr, P. (2022). Gradients without backpropagation. arXiv preprint arXiv:2202.08587.
    """
    PRE_MULTIPLY_BY_H = False
    def __init__(
        self,
        n_samples: int = 1,
        distribution: Distributions = "gaussian",
        pre_generate = True,
        jvp_method: Literal['autograd', 'forward', 'central'] = 'autograd',
        h: float = 1e-3,
        target: GradTarget = "closure",
        seed: int | None | torch.Generator = None,
    ):
        super().__init__(h=h, n_samples=n_samples, distribution=distribution, target=target, pre_generate=pre_generate, seed=seed)
        self.defaults['jvp_method'] = jvp_method

    @torch.no_grad
    def approximate(self, closure, params, loss):
        params = TensorList(params)
        loss_approx = None

        fs = self.settings[params[0]]
        n_samples = fs['n_samples']
        jvp_method = fs['jvp_method']
        h = fs['h']
        distribution = fs['distribution']
        default = [None]*n_samples
        perturbations = list(zip(*(self.state[p].get('perturbations', default) for p in params)))
        generator = self.get_generator(params[0].device, self.defaults['seed'])

        grad = None
        for i in range(n_samples):
            prt = perturbations[i]
            if prt[0] is None:
                prt = params.sample_like(distribution=distribution, variance=1, generator=generator)

            else: prt = TensorList(prt)

            if jvp_method == 'autograd':
                with torch.enable_grad():
                    loss, d = jvp(partial(closure, False), params=params, tangent=prt)

            elif jvp_method == 'forward':
                loss, d = jvp_fd_forward(partial(closure, False), params=params, tangent=prt, v_0=loss, h=h)

            elif jvp_method == 'central':
                loss_approx, d = jvp_fd_central(partial(closure, False), params=params, tangent=prt, h=h)

            else: raise ValueError(jvp_method)

            if grad is None: grad = prt * d
            else: grad += prt * d

        assert grad is not None
        if n_samples > 1: grad.div_(n_samples)
        return grad, loss, loss_approx

