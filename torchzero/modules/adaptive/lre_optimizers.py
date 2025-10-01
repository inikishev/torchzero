"""subspace optimizers to be used in a low rank eigenbasis

three opts support this - LMAdagrad and experimental AdaNystrom and Eigengrad

I could define repoject on a module but because most opts use per-parameter state that is complicated"""

import math
from abc import ABC, abstractmethod

import torch

from ...linalg import matrix_power_eigh


class LREOptimizerBase(ABC):
    """Optimizer to run in a low rank eigenbasis.

    notes:

    1. it shouldn't store any states in self, everything should be in state.
    This is because this may be called on multiple parameters in a sequence

    2. apply is always called first, than reproject whenever eigenbasis gets updated

    3. L is variance in the eigenbasis.
    """
    @abstractmethod
    def step(self, g: torch.Tensor, L: torch.Tensor, Q: torch.Tensor, state: dict) -> torch.Tensor:
        ...

    @abstractmethod
    def reproject(self, L_old: torch.Tensor, Q_old: torch.Tensor,
                  L_new: torch.Tensor, Q_new: torch.Tensor, state: dict) -> None:
        ...

class Whiten(LREOptimizerBase):
    """This simply applies whitening and is equivalent to not running an optimizer in the eigenbasis"""
    def step(self, g, L, Q, state): return (Q * L.rsqrt()) @ (Q.T @ g)
    def reproject(self, L_old, Q_old, L_new, Q_new, state): pass

class EMA(LREOptimizerBase):
    """Maintains exponential moving average of gradients in the low rank eigenbasis. Nesterov setting is experimental"""
    def __init__(self, beta=0.9, nesterov:bool=False):
        self.beta = beta
        self.nesterov = nesterov

    def step(self, g, L, Q, state):
        g = Q.T @ g

        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(g)

        exp_avg = state["exp_avg"]
        exp_avg.lerp_(g, 1-self.beta)

        if self.nesterov:
            dir = (g + exp_avg * self.beta) / (1 + self.beta)
        else:
            dir = exp_avg

        return (Q * L.rsqrt()) @ dir

    def reproject(self, L_old, Q_old, L_new, Q_new, state):
        if  "exp_avg" not in state: return
        state["exp_avg"] = Q_new.T @ (Q_old @ state["exp_avg"])

class Adam(LREOptimizerBase):
    """Runs Adam in low rank eigenbasis."""
    def __init__(self, beta1=0.9, beta2=0.95, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def step(self, g, L, Q, state):
        g = Q.T @ g

        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(g)
            state["exp_avg_sq"] = torch.zeros_like(g)
            state["current_step"] = 1

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        current_step = state["current_step"]

        exp_avg.lerp_(g, 1-self.beta1)
        exp_avg_sq.mul_(self.beta2).addcmul_(g, g, value=1-self.beta2)
        denom = exp_avg_sq.sqrt().add_(self.eps)

        bias_correction1 = 1.0 - (self.beta1 ** current_step)
        bias_correction2 = 1.0 - (self.beta2 ** current_step)
        alpha = math.sqrt(bias_correction2) / bias_correction1
        state["current_step"] = current_step + 1

        dir = (exp_avg * alpha) / denom
        return Q @ dir

    def reproject(self, L_old, Q_old, L_new, Q_new, state):
        if  "exp_avg" not in state: return
        C = Q_new.T @ Q_old

        state["exp_avg"] = C @ state["exp_avg"]
        state["exp_avg_sq"] = C.square() @ state["exp_avg_sq"]


class FullMatrixAdam(LREOptimizerBase):
    """Runs full-matrix Adam in low rank eigenbasis.
    The preconditioner is updated whenever basis is updated"""
    def __init__(self, beta1=0.9, beta2=0.95, eps=1e-8, matrix_power=-1/2, abs=True):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.matrix_power = matrix_power
        self.abs = abs

    def step(self, g, L, Q, state):
        g = Q.T @ g

        # initialize
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(g)
            state["covariance"] = torch.eye(g.numel(), device=g.device, dtype=g.dtype)
            state["preconditioner"] = torch.eye(g.numel(), device=g.device, dtype=g.dtype)
            state["reprojected"] = True
            state["current_step"] = 1

        exp_avg = state["exp_avg"]
        covariance = state["covariance"]
        current_step = state["current_step"]

        # update buffers
        exp_avg.lerp_(g, 1-self.beta1)
        covariance.lerp_(g.outer(g), weight=1-self.beta2)

        # correct bias
        bias_correction1 = 1.0 - (self.beta1 ** current_step)
        exp_avg = exp_avg / bias_correction1

        # after reprojecting update the preconditioner
        if state["reprojected"]:
            state["reprojected"] = False

            bias_correction2 = 1.0 - (self.beta2 ** current_step)
            covariance = covariance / bias_correction2

            reg = torch.eye(covariance.size(0), device=covariance.device, dtype=covariance.dtype).mul_(self.eps)
            covariance = covariance + reg

            # compute matrix power
            try:
                state["preconditioner"] = matrix_power_eigh(covariance, self.matrix_power, abs=self.abs)

            except torch.linalg.LinAlgError:

                # fallback to diagonal
                state["preconditioner"] = covariance.diagonal().rsqrt().diag_embed()

        # compute the update
        state["current_step"] = current_step + 1
        preconditioner = state["preconditioner"]
        dir = preconditioner @ exp_avg
        return Q @ dir

    def reproject(self, L_old, Q_old, L_new, Q_new, state):
        if  "exp_avg" not in state: return

        state["reprojected"] = True

        C = Q_new.T @ Q_old
        state["exp_avg"] = C @ state["exp_avg"]
        state["covariance"] = C @ state["covariance"] @ C.T
