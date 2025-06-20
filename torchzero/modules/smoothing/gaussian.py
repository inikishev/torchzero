import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal

import torch

from ...core import Modular, Module, Var
from ...utils import NumberList, TensorList
from ...utils.derivatives import jacobian_wrt
from ..grad_approximation import GradApproximator, GradTarget


class Reformulation(Module, ABC):
    def __init__(self, defaults):
        super().__init__(defaults)

    @abstractmethod
    def closure(self, backward: bool, closure: Callable, params:list[torch.Tensor], var: Var) -> tuple[float | torch.Tensor, Sequence[torch.Tensor] | None]:
        """returns loss and gradient, if backward is False then gradient can be None"""

    def pre_step(self, var: Var) -> Var | None:
        """This runs once before each step, whereas `closure` may run multiple times per step if further modules
        evaluate gradients at multiple points. This is useful for example to pre-generate new random perturbations."""
        return var

    def step(self, var):
        ret = self.pre_step(var)
        if isinstance(ret, Var): var = ret

        if var.closure is None: raise RuntimeError("Reformulation requires closure")
        params, closure = var.params, var.closure


        def modified_closure(backward=True):
            loss, grad = self.closure(backward, closure, params, var)

            if grad is not None:
                for p,g in zip(params, grad):
                    p.grad = g

            return loss

        var.closure = modified_closure
        return var


def _decay_sigma_(self: Module, params):
    for p in params:
        state = self.state[p]
        settings = self.settings[p]
        state['sigma'] *= settings['decay']

def _generate_perturbations_to_state_(self: Module, params: TensorList, n_samples, sigmas, generator):
    perturbations = [params.sample_like(generator=generator) for _ in range(n_samples)]
    torch._foreach_mul_([p for l in perturbations for p in l], [v for vv in sigmas for v in [vv]*n_samples])
    for param, prt in zip(params, zip(*perturbations)):
        self.state[param]['perturbations'] = prt

def _clear_state_hook(optimizer: Modular, var: Var, self: Module):
    for m in optimizer.unrolled_modules:
        if m is not self:
            m.reset()

class GaussianHomotopy(Reformulation):
    """Approximately smoothes the function with a gaussian kernel by sampling it at random perturbed points around current point. Both function values and gradients are averaged over all samples. The perturbed points are generated before each
    step and remain the same throughout the step.

    .. note::
        This module reformulates the objective, it modifies the closure to evaluate value and gradients of a smoothed function. All modules after this will operate on the modified objective.

    .. note::
        This module requires the a closure passed to the optimizer step,
        as it needs to re-evaluate the loss and gradients at perturbed points.

    Args:
        n_samples (int): number of points to sample, larger values lead to a more accurate smoothing.
        init_sigma (float): initial scale of perturbations.
        tol (float | None, optional):
            if maximal parameters change value is smaller than this, sigma is reduced by :code:`decay`. Defaults to 1e-4.
        decay (float, optional): multiplier to sigma when converged on a smoothed function. Defaults to 0.5.
        max_steps (int | None, optional): maximum number of steps before decaying sigma. Defaults to None.
        clear_state (bool, optional):
            whether to clear all other module states when sigma is decayed, because the objective function changes. Defaults to True.
        seed (int | None, optional): seed for random perturbationss. Defaults to None.

    Examples:
        Gaussian-smoothed NewtonCG

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.GaussianHomotopy(100),
                tz.m.NewtonCG(maxiter=20),
                tz.m.AdaptiveBacktracking(),
            )

    """
    def __init__(
        self,
        n_samples: int,
        init_sigma: float,
        tol: float | None = 1e-4,
        decay=0.5,
        max_steps: int | None = None,
        clear_state=True,
        seed: int | None = None,
    ):
        defaults = dict(n_samples=n_samples, init_sigma=init_sigma, tol=tol, decay=decay, max_steps=max_steps, clear_state=clear_state, seed=seed)
        super().__init__(defaults)


    def _get_generator(self, seed: int | None | torch.Generator, params: list[torch.Tensor]):
        if 'generator' not in self.global_state:
            if isinstance(seed, torch.Generator): self.global_state['generator'] = seed
            elif seed is not None: self.global_state['generator'] = torch.Generator(params[0].device).manual_seed(seed)
            else: self.global_state['generator'] = None
        return self.global_state['generator']

    def pre_step(self, var):
        params = TensorList(var.params)
        settings = self.settings[params[0]]
        n_samples = settings['n_samples']
        init_sigma = [self.settings[p]['init_sigma'] for p in params]
        sigmas = self.get_state(params, 'sigma', init=init_sigma)

        if any('perturbations' not in self.state[p] for p in params):
            generator = self._get_generator(settings['seed'], params)
            _generate_perturbations_to_state_(self, params=params, n_samples=n_samples, sigmas=sigmas, generator=generator)

        # sigma decay rules
        max_steps = settings['max_steps']
        decayed = False
        if max_steps is not None and max_steps > 0:
            level_steps = self.global_state['level_steps'] = self.global_state.get('level_steps', 0) + 1
            if level_steps > max_steps:
                self.global_state['level_steps'] = 0
                _decay_sigma_(self, params)
                decayed = True

        tol = settings['tol']
        if tol is not None and not decayed:
            if not any('prev_params' in self.state[p] for p in params):
                prev_params = self.get_state(params, 'prev_params', cls=TensorList, init='param')
            else:
                prev_params = self.get_state(params, 'prev_params', cls=TensorList, init='param')
                s = params - prev_params

                if s.abs().global_max() <= tol:
                    _decay_sigma_(self, params)
                    decayed = True

                prev_params.copy_(params)

        if decayed:
            generator = self._get_generator(settings['seed'], params)
            _generate_perturbations_to_state_(self, params=params, n_samples=n_samples, sigmas=sigmas, generator=generator)
            if settings['clear_state']:
                var.post_step_hooks.append(partial(_clear_state_hook, self=self))

    @torch.no_grad
    def closure(self, backward, closure, params, var):
        params = TensorList(params)

        settings = self.settings[params[0]]
        n_samples = settings['n_samples']

        perturbations = list(zip(*(self.state[p]['perturbations'] for p in params)))

        loss = None
        grad = None
        for i in range(n_samples):
            prt = perturbations[i]

            params.add_(prt)
            if backward:
                with torch.enable_grad(): l = closure()
                if grad is None: grad = params.grad
                else: grad += params.grad

            else:
                l = closure(False)

            if loss is None: loss = l
            else: loss = loss+l

            params.sub_(prt)

        assert loss is not None
        if n_samples > 1:
            loss = loss / n_samples
            if backward:
                assert grad is not None
                grad.div_(n_samples)

        return loss, grad
