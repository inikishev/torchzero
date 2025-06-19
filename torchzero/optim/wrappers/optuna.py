import typing
from collections import abc

import numpy as np
import torch

import optuna

from ...utils import Optimizer

def silence_optuna():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

def _ensure_float(x) -> float:
    if isinstance(x, torch.Tensor): return x.detach().cpu().item()
    if isinstance(x, np.ndarray): return float(x.item())
    return float(x)


class OptunaSampler(Optimizer):
    """Optimize your next SOTA model using hyperparameter optimization.

    Note - optuna is surprisingly scalable to large number of parameters (up to 10,000), despite literally requiring a for-loop because it only supports scalars. Default TPESampler is good for BBO. Maybe not for NNs...

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lb (float): lower bounds.
        ub (float): upper bounds.
        sampler (optuna.samplers.BaseSampler | type[optuna.samplers.BaseSampler] | None, optional): sampler. Defaults to None.
        silence (bool, optional): makes optuna not write a lot of very useful information to console. Defaults to True.
    """
    def __init__(
        self,
        params,
        lb: float,
        ub: float,
        sampler: "optuna.samplers.BaseSampler | type[optuna.samplers.BaseSampler] | None" = None,
        silence: bool = True,
    ):
        if silence: silence_optuna()
        super().__init__(params, lb=lb, ub=ub)

        if isinstance(sampler, type): sampler = sampler()
        self.sampler = sampler
        self.study = None

    @torch.no_grad
    def step(self, closure):

        params = self.get_params()
        if self.study is None:
            self.study = optuna.create_study(sampler=self.sampler)

        # some optuna samplers use torch
        with torch.enable_grad():
            trial = self.study.ask()

            suggested = []
            for gi,g in enumerate(self.param_groups):
                for pi,p in enumerate(g['params']):
                    lb, ub =  g['lb'], g['ub']
                    suggested.extend(trial.suggest_float(f'g{gi}_p{pi}_w{i}', lb, ub) for i in range(p.numel()))

        vec = torch.as_tensor(suggested).to(params[0])
        params.from_vec_(vec)

        loss = closure()
        with torch.enable_grad(): self.study.tell(trial, loss)

        return loss