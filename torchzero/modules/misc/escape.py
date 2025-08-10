import math

import torch

from ...core import Modular, Module, Var, Chainable
from ...utils import NumberList, TensorList


class EscapeAnnealing(Module):
    """If parameters stop changing, this runs a backward annealing random search"""
    def __init__(self, max_region:float = 1, max_iter:int = 1000, tol=1e-6, n_tol: int = 10):
        defaults = dict(max_region=max_region, max_iter=max_iter, tol=tol, n_tol=n_tol)
        super().__init__(defaults)


    @torch.no_grad
    def step(self, var):
        closure = var.closure
        if closure is None: raise RuntimeError("Escape requries closure")

        params = TensorList(var.params)
        settings = self.settings[params[0]]
        max_region = self.get_settings(params, 'max_region', cls=NumberList)
        max_iter = settings['max_iter']
        tol = settings['tol']
        n_tol = settings['n_tol']

        n_bad = self.global_state.get('n_bad', 0)

        prev_params = self.get_state(params, 'prev_params', cls=TensorList)
        diff = params-prev_params
        prev_params.copy_(params)

        if diff.abs().global_max() <= tol:
            n_bad += 1

        else:
            n_bad = 0

        self.global_state['n_bad'] = n_bad

        # no progress
        f_0 = var.get_loss(False)
        if n_bad >= n_tol:
            for i in range(1, max_iter+1):
                alpha = max_region * (i / max_iter)
                pert = params.sphere_like(radius=alpha)

                params.add_(pert)
                f_star = closure(False)

                if math.isfinite(f_star) and f_star < f_0-1e-10:
                    var.update = None
                    var.stop = True
                    var.skip_update = True
                    return var

                params.sub_(pert)

            self.global_state['n_bad'] = 0
        return var

class ResetOnStuck(Module):
    """Resets optimizer state when update (difference in parameters) is close to zero for multiple steps in a row."""
    def __init__(self, modules: Chainable, tol=1e-10, n_tol: int = 4):
        defaults = dict(tol=tol, n_tol=n_tol)
        super().__init__(defaults)
        self.set_child('modules', modules)

    @torch.no_grad
    def update(self, var):
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        params = TensorList(var.params)
        settings = self.settings[params[0]]
        tol = settings['tol']
        n_tol = settings['n_tol']
        n_bad = self.global_state.get('n_bad', 0)
        modules = self.children['modules']

        # calculate difference in parameters
        prev_params = self.get_state(params, 'prev_params', cls=TensorList)
        update = params - prev_params
        prev_params.copy_(params)

        # if update is too small, it is considered bad, otherwise n_bad is reset to 0
        if step > 0:
            if update.abs().global_max() <= tol:
                n_bad += 1

            else:
                n_bad = 0

        self.global_state['n_bad'] = n_bad

        # no progress, reset
        if n_bad >= n_tol:
            modules.reset()
            self.global_state['n_bad'] = 0
            self.global_state['step'] = 0

        # step with child (after resetting if reset)
        modules.update(var)

    @torch.no_grad
    def apply(self, var):
        modules = self.children['modules']
        return modules.apply(var.clone(clone_update=False))

    def get_H(self, var):
        return self.children['modules'].get_H(var)