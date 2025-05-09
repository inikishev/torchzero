import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal

import torch

from ...core import Modular, Module, Vars
from ...utils import NumberList, TensorList
from ...utils.derivatives import jacobian_wrt
from ..grad_approximation import GradApproximator, GradTarget


class Reformulation(Module, ABC):
    def __init__(self, defaults):
        super().__init__(defaults)

    @abstractmethod
    def closure(self, backward: bool, closure: Callable, params:list[torch.Tensor], vars: Vars) -> tuple[float | torch.Tensor, Sequence[torch.Tensor] | None]:
        """returns loss and gradient, if backward is False then gradient can be None"""

    def pre_step(self, vars: Vars) -> Vars | None:
        """This runs once before each step, whereas `closure` may run multiple times per step if further modules
        evaluate gradients at multiple points. This is useful for example to pre-generate new random perturbations."""
        return vars

    def step(self, vars):
        ret = self.pre_step(vars)
        if isinstance(ret, Vars): vars = ret

        if vars.closure is None: raise RuntimeError("Reformulation requires closure")
        params, closure = vars.params, vars.closure


        def modified_closure(backward=True):
            loss, grad = self.closure(backward, closure, params, vars)

            if grad is not None:
                for p,g in zip(params, grad):
                    p.grad = g

            return loss

        vars.closure = modified_closure
        return vars


def _decay_sigma_(self: Module, params):
    for p in params:
        state = self.state[p]
        settings = self.settings[p]
        state['sigma'] *= settings['decay']

def _generate_perturbations_to_state_(self: Module, params: TensorList, n_samples, sigmas):
    perturbations = [params.sample_like() for _ in range(n_samples)]
    torch._foreach_mul_([p for l in perturbations for p in l], [v for vv in sigmas for v in [vv]*n_samples])
    for param, prt in zip(params, zip(*perturbations)):
        self.state[param]['perturbations'] = prt

def _clear_state_hook(optimizer: Modular, vars: Vars, self: Module):
    for m in optimizer.unrolled_modules:
        if m is not self:
            m.reset()

class GaussianHomotopy(Reformulation):
    def __init__(
        self,
        n_samples: int,
        init_sigma: float,
        tol: float | None = 1e-4,
        decay=0.5,
        max_steps: int | None = None,
        clear_state=True,
    ):
        defaults = dict(n_samples=n_samples, init_sigma=init_sigma, tol=tol, decay=decay, max_steps=max_steps, clear_state=clear_state)
        super().__init__(defaults)

    def pre_step(self, vars):
        params = TensorList(vars.params)
        settings = self.settings[params[0]]
        n_samples = settings['n_samples']
        init_sigma = self.get_settings('init_sigma', params=params)
        sigmas = self.get_state('sigma', params = params, init=init_sigma)

        if any('perturbations' not in self.state[p] for p in params):
            _generate_perturbations_to_state_(self, params, n_samples, sigmas)

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
                prev_params = self.get_state('prev_params', params=params, cls=TensorList, init='param')
            else:
                prev_params = self.get_state('prev_params', params=params, cls=TensorList, init='param')
                s = params - prev_params

                if s.abs().global_max() <= tol:
                    _decay_sigma_(self, params)
                    decayed = True

                prev_params.copy_(params)

        if decayed:
            _generate_perturbations_to_state_(self, params, n_samples, sigmas)
            if settings['clear_state']:
                vars.post_step_hooks.append(partial(_clear_state_hook, self=self))

    @torch.no_grad
    def closure(self, backward, closure, params, vars):
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
