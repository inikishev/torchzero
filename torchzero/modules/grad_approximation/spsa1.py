from collections.abc import Callable
from typing import Any
from functools import partial
import torch

from ...utils import TensorList, NumberList
from ..grad_approximation.grad_approximator import GradApproximator, GradTarget

class SPSA1(GradApproximator):
    """One-measurement variant of SPSA. Unlike standard two-measurement SPSA, the estimated
    gradient often won't be a descent direction, however the expectation is biased towards
    the descent direction. Therefore this variant of SPSA is only recommended for a specific
    class of problems where the objective function changes on each evaluation,
    for example feedback control problems.

    Args:
        h (float, optional):
            finite difference step size, recommended to set to same value as learning rate. Defaults to 1e-3.
        n_samples (int, optional): number of random samples. Defaults to 1.
        eps (float, optional): measurement noise estimate. Defaults to 1e-8.
        seed (int | None | torch.Generator, optional): random seed. Defaults to None.
        target (GradTarget, optional): what to set on closure. Defaults to "closure".

    Reference:
        [SPALL, JAMES C. "A One-measurement Form of Simultaneous Stochastic Approximation](https://www.jhuapl.edu/spsa/PDF-SPSA/automatica97_one_measSPSA.pdf)."
    """

    def __init__(
        self,
        h: float = 1e-3,
        n_samples: int = 1,
        eps: float = 1e-8, # measurement noise
        pre_generate = False,
        seed: int | None | torch.Generator = None,
        target: GradTarget = "closure",
    ):
        defaults = dict(h=h, eps=eps, n_samples=n_samples, pre_generate=pre_generate, seed=seed)
        super().__init__(defaults, target=target)


    def pre_step(self, objective):

        if self.defaults['pre_generate']:

            params = TensorList(objective.params)
            generator = self.get_generator(params[0].device, self.defaults['seed'])

            n_samples = self.defaults['n_samples']
            h = self.get_settings(objective.params, 'h')

            perturbations = [params.rademacher_like(generator=generator) for _ in range(n_samples)]
            torch._foreach_mul_([p for l in perturbations for p in l], [v for vv in h for v in [vv]*n_samples])

            for param, prt in zip(params, zip(*perturbations)):
                self.state[param]['perturbations'] = prt

    @torch.no_grad
    def approximate(self, closure, params, loss):
        generator = self.get_generator(params[0].device, self.defaults['seed'])

        params = TensorList(params)
        orig_params = params.clone() # store to avoid small changes due to float imprecision
        loss_approx = None

        h, eps = self.get_settings(params, "h", "eps", cls=NumberList)
        n_samples = self.defaults['n_samples']

        default = [None]*n_samples
        # perturbations are pre-multiplied by h
        perturbations = list(zip(*(self.state[p].get('perturbations', default) for p in params)))

        grad = None
        for i in range(n_samples):
            prt = perturbations[i]

            if prt[0] is None:
                prt = params.rademacher_like(generator=generator).mul_(h)

            else: prt = TensorList(prt)

            params += prt
            L = closure(False)
            params.copy_(orig_params)

            sample = prt * ((L + eps) / h)
            if grad is None: grad = sample
            else: grad += sample

        assert grad is not None
        if n_samples > 1: grad.div_(n_samples)

        # mean if got per-sample values
        return grad, loss, loss_approx
