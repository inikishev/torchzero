from contextlib import nullcontext

import numpy as np
import torch

from ... import tl
from ...utils.python_tools import _ScalarLoss
from ...core import _ClosureType, OptimizationState, OptimizerModule


def _numpy_or_torch_mean(losses: list):
    """Returns the mean of a list of losses, which can be either numpy arrays or torch tensors."""
    if isinstance(losses[0], torch.Tensor):
        return torch.mean(torch.stack(losses))
    return np.mean(losses).item()

class ApproxGaussianSmoothing(OptimizerModule):
    """Samples and averages value and gradients in multiple random points around current position.
    This effectively applies smoothing to the function.

    Args:
        n_samples (int, optional): number of gradient samples from around current position. Defaults to 4.
        sigma (float, optional): how far from current position to sample from. Defaults to 0.1.
        distribution (tl.Distributions, optional): distribution for random positions. Defaults to "normal".
        sample_x0 (bool, optional): 1st sample will be x0. Defaults to False.
        randomize_every (int | None, optional): randomizes the points every n steps. Defaults to 1.
    """
    def __init__(
        self,
        n_samples: int = 4,
        sigma: float = 0.1,
        distribution: tl.Distributions = "normal",
        sample_x0 = False,
        randomize_every: int | None = 1,
    ):
        defaults = dict(sigma = sigma)
        super().__init__(defaults)
        self.n_samples = n_samples
        self.distribution: tl.Distributions = distribution
        self.randomize_every = randomize_every
        self.current_step = 0
        self.perturbations = None
        self.sample_x0 = sample_x0

    @torch.no_grad()
    def step(self, state: OptimizationState):
        if state.closure is None: raise ValueError('GaussianSmoothing requires closure.')
        closure = state.closure
        params = self.get_params()
        sigmas = self.get_group_key('sigma')

        # generate random perturbations
        if self.perturbations is None or (self.randomize_every is not None and self.current_step % self.randomize_every == 0):
            if self.sample_x0:
                self.perturbations = [params.sample_like(sigmas, distribution=self.distribution) for _ in range(self.n_samples-1)]
            else:
                self.perturbations = [params.sample_like(sigmas, distribution=self.distribution) for _ in range(self.n_samples)]

        @torch.no_grad
        def smooth_closure(backward = True, **k):
            losses = []
            grads = []

            # sample gradient and loss at x0
            if self.sample_x0:
                with torch.enable_grad() if backward else nullcontext():
                    losses.append(closure(backward, **k))
                    if backward: grads.append(params.grad.clone())

            # sample gradients from points around current params
            # and average them
            if self.perturbations is None: raise ValueError('who set perturbations to None???')
            for p in self.perturbations:
                params.add_(p)
                with torch.enable_grad() if backward else nullcontext():
                    losses.append(closure(backward, **k))
                    if backward: grads.append(params.grad.clone())
                params.sub_(p)

            # set the new averaged grads and return average loss
            if backward: params.set_grad_(tl.mean(grads))
            return _numpy_or_torch_mean(losses)


        self.current_step += 1
        state.closure = smooth_closure
        return self._update_params_or_step_with_next(state)