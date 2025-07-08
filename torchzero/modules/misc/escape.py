import torch

from ...core import Module
from ...utils import TensorList, NumberList


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
                pert = params.sample_like(distribution='sphere').mul_(alpha)

                params.add_(pert)
                f_star = closure(False)

                if f_star < f_0-1e-10:
                    var.update = None
                    var.stop = True
                    var.skip_update = True
                    return var

                else:
                    params.sub_(pert)

            self.global_state['n_bad'] = 0
        return var