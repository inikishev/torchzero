import math
from abc import ABC, abstractmethod
from collections import deque

import torch

from ...core import Chainable, TensorTransform
from ...linalg import torch_linalg
from ...linalg.matrix_power import matrix_power_eigh
from ...modules.adaptive.lmadagrad import lm_adagrad_update
from ...modules.quasi_newton.quasi_newton import bfgs_B_, bfgs_H_
from ...utils import NumberList, TensorList
from .cubic_adam import cubic_adam_

def debiased_step_size(step, beta1, beta2):
    bias_correction1 = 1.0 - (beta1 ** step)
    bias_correction2 = 1.0 - (beta2 ** step)
    return math.sqrt(bias_correction2) / bias_correction1

class SubspaceOptimizerBase(ABC):
    @abstractmethod
    def step(self, g: torch.Tensor, U: torch.Tensor, L: torch.Tensor, state: dict) -> torch.Tensor:
        ...

    @abstractmethod
    def reproject(self, U_old: torch.Tensor, L_old: torch.Tensor,
                  U_new: torch.Tensor, L_new: torch.Tensor, state: dict) -> None:
        ...

class SubspaceWhiten(SubspaceOptimizerBase):
    """To be used within ``tz.m.experimental.LMSubspace``. This simply applies whitening and is equivalent to ``tz.m.LMAdagrad``"""
    def step(self, g, U, L, state): return (U * L.rsqrt()) @ g
    def reproject(self, U_old, L_old, U_new, L_new, state): pass

class SubspaceMomentum(SubspaceOptimizerBase):
    """To be used within ``tz.m.experimental.LMSubspace``. Maintains exponential moving average of gradients,
    equivalent to ``beta`` setting in ``tz.m.LMAdagrad``. Nesterov setting is experimental"""
    def __init__(self, beta=0.9, nesterov:bool=False):
        self.beta = beta
        self.nesterov = nesterov

    def step(self, g, U, L, state):
        if "exp_avg_proj" not in state:
            state["exp_avg_proj"] = torch.zeros_like(g)

        exp_avg = state["exp_avg_proj"]
        exp_avg.lerp_(g, 1-self.beta)

        if self.nesterov:
            dir = (g + exp_avg * self.beta) / (1 + self.beta)
        else:
            dir = exp_avg

        return (U * L.rsqrt()) @ dir

    def reproject(self, U_old, L_old, U_new, L_new, state):
        if  "exp_avg_proj" not in state: return
        state["exp_avg_proj"] = U_new.T @ (U_old @ state["exp_avg_proj"])

class SubspaceAdam(SubspaceOptimizerBase):
    """To be used within ``tz.m.experimental.LMSubspace``. Runs Adam in the eigenbasis of LMAdagrad."""
    def __init__(self, beta1=0.9, beta2=0.95, eps=1e-8):
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        self.current_step = 1

    def step(self, g, U, L, state):
        if "exp_avg_proj" not in state:
            state["exp_avg_proj"] = torch.zeros_like(g)
            state["exp_avg_sq_proj"] = torch.zeros_like(g)

        exp_avg = state["exp_avg_proj"]
        exp_avg_sq = state["exp_avg_sq_proj"]

        exp_avg.lerp_(g, 1-self.beta1)
        exp_avg_sq.mul_(self.beta2).addcmul_(g,g, value=1-self.beta2)

        alpha = debiased_step_size(self.current_step, beta1=self.beta1, beta2=self.beta2)
        denom = exp_avg_sq.sqrt().add_(self.eps)

        self.current_step += 1
        dir = (exp_avg * alpha) / denom
        return U @ dir

    def reproject(self, U_old, L_old, U_new, L_new, state):
        if  "exp_avg_proj" not in state: return

        C = U_new.T @ U_old

        state["exp_avg_proj"] = C @ state["exp_avg_proj"]
        state["exp_avg_sq_proj"] = C.square() @ state["exp_avg_sq_proj"]

class SubspaceCubicAdam(SubspaceOptimizerBase):
    """To be used within ``tz.m.experimental.LMSubspace``. Runs cubic Adam (see ``tz.m.experimental.CubicAdam``) in the eigenbasis of LMAdagrad."""
    def __init__(self, beta1=0.9, beta2=0.95, beta3=0.95, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.eps = eps
        self.current_step = 1

    def step(self, g, U, L, state):
        if "exp_avg_proj" not in state:
            state["exp_avg_proj"] = torch.zeros_like(g)
            state["exp_avg_sq_proj"] = torch.zeros_like(g)
            state["exp_avg_cu_proj"] = torch.zeros_like(g)

        dir = cubic_adam_(
            tensors = TensorList([g]),
            exp_avg_ = TensorList([state["exp_avg_proj"]]),
            exp_avg_sq_ = TensorList([state["exp_avg_sq_proj"]]),
            exp_avg_cu_ = TensorList([state["exp_avg_cu_proj"]]),
            alpha = 1,
            beta1 = self.beta1,
            beta2 = self.beta2,
            beta3 = self.beta3,
            eps = self.eps,
            debiased = True,
            step = self.current_step,
        )[0]

        return U @ dir

    def reproject(self, U_old, L_old, U_new, L_new, state):
        if  "exp_avg_proj" not in state: return

        C = U_new.T @ U_old

        state["exp_avg_proj"] = C @ state["exp_avg_proj"]
        state["exp_avg_sq_proj"] = C.square() @ state["exp_avg_sq_proj"]
        state["exp_avg_cu_proj"] = C.pow(3) @ state["exp_avg_cu_proj"]


class SubspaceFullMatrixAdam(SubspaceOptimizerBase):
    """To be used within ``tz.m.experimental.LMSubspace``. Runs full-matrix Adam in the eigenbasis of LMAdagrad."""
    def __init__(self, beta1=0.95, beta2=0.95, eps=1e-8, matrix_power=-1/2, abs=abs):
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        self.current_step = 1
        self.matrix_power = matrix_power
        self.abs = abs

    def step(self, g, U, L, state):
        if "exp_avg_proj" not in state:
            state["exp_avg_proj"] = torch.zeros_like(g)
            state["covariance"] = torch.eye(g.numel(), device=g.device, dtype=g.dtype)

        exp_avg = state["exp_avg_proj"]
        covariance = state["covariance"]

        exp_avg.lerp_(g, 1-self.beta1)
        covariance.lerp_(g.outer(g), weight=1-self.beta2)

        alpha = debiased_step_size(self.current_step, beta1=self.beta1, beta2=self.beta2)
        reg = torch.eye(g.numel(), device=g.device, dtype=g.dtype).mul_(self.eps)

        try:
            P = matrix_power_eigh(covariance + reg, self.matrix_power, abs=self.abs)
        except torch.linalg.LinAlgError:
            return U @ exp_avg.clip(-0.1, 0.1)

        self.current_step += 1
        dir = P @ (exp_avg * alpha)
        return U @ dir

    def reproject(self, U_old, L_old, U_new, L_new, state):
        if  "exp_avg_proj" not in state: return

        C = U_new.T @ U_old

        # reproject first moments
        state["exp_avg_proj"] = C @ state["exp_avg_proj"]

        # reproject covariance
        state["covariance"] = C @ state["covariance"] @ C.T


def opt_lm_adagrad_apply(g: torch.Tensor, U: torch.Tensor, L: torch.Tensor, opt: SubspaceOptimizerBase, state:dict):
    z = U.T @ g
    return opt.step(g=z, U=U, L=L, state=state)


def maybe_lerp(cur: torch.Tensor | None, new: torch.Tensor | None, beta: float | None):
    if cur is None: return new
    if new is None: return cur
    if beta is None or beta == 0: return new
    return cur.lerp(new, weight=1-beta)

class LMSubspace(TensorTransform):
    """
    Use a subspace optimizer in LMAdagrad's eigenbasis (any subclass of ``SubspaceOptimizerBase``).

    The eigenbasis is of size ``history_size`` variables, usually smaller because small eigenvalues are removed, therefore we can run expensive update rules such as full-matrix whitening. The eigenbasis is already rotated to be diagonal though so whether full-matrix preconditioning is useful is unclear.

    The subspace optimizer buffers are reprojected whenever factors are updated (this is why we need special class which defines ``reproject`` method).

    Example:
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.experimental.LMSubspace(tz.m.experimental.SubspaceFullMatrixAdam()),
        tz.m.LR(1e-2),
    ```
    )
    """
    def __init__(
        self,
        subspace_opt: SubspaceOptimizerBase,
        history_size: int = 100,
        update_freq: int = 1,
        damping: float = 1e-4,
        rdamping: float = 0,
        tol=1e-7,
        truncate: int | None = None,
        order: int = 1,
        true_damping: bool = True,
        U_beta: float | None = None,
        L_beta: float | None = None,
        concat_params: bool = True,

        inner: Chainable | None = None,
        U_tfm: Chainable | None = None,
        L_tfm: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner'], defaults['concat_params'], defaults["U_tfm"], defaults["L_tfm"]

        super().__init__(defaults, concat_params=concat_params, inner=inner)

        self.set_child("U", U_tfm)
        self.set_child("L", L_tfm)

        # need to handle saving/loading this
        # or just move to global state and turn into property
        self.subspace_opt = subspace_opt


    @torch.no_grad
    def single_tensor_update(self, tensor, param, grad, loss, state, setting):
        order = setting['order']
        history_size = setting['history_size']
        update_freq = setting['update_freq']
        U_beta = setting['U_beta']
        L_beta = setting['L_beta']

        if 'history' not in state: state['history'] = deque(maxlen=history_size)
        history = state['history']

        if order == 1:
            t = tensor.clone().view(-1)
            history.append(t)
        else:

            # if order=2, history is of gradient differences, order 3 is differences between differences, etc
            # scaled by parameter differences
            cur_p = param.clone()
            cur_g = tensor.clone()
            eps = torch.finfo(cur_p.dtype).tiny * 2
            for i in range(1, order):
                if f'prev_g_{i}' not in state:
                    state[f'prev_p_{i}'] = cur_p
                    state[f'prev_g_{i}'] = cur_g
                    break

                s = cur_p - state[f'prev_p_{i}']
                y = cur_g - state[f'prev_g_{i}']
                state[f'prev_p_{i}'] = cur_p
                state[f'prev_g_{i}'] = cur_g
                cur_p = s
                cur_g = y

                if i == order - 1:
                    cur_g = cur_g / torch.linalg.norm(cur_p).clip(min=eps) # pylint:disable=not-callable
                    history.append(cur_g.view(-1))

        step = state.get('step', 0)
        if step % update_freq == 0 and len(history) != 0:

            # update factors
            U, L = lm_adagrad_update(
                history,
                damping=setting["damping"],
                rdamping=setting["rdamping"],
                truncate=setting["truncate"],
                tol=setting["tol"],
            )

            U_old = state.get("U", None)
            L_old = state.get("L", None)

            L_new = maybe_lerp(L_old, L, L_beta)
            U_new = maybe_lerp(U_old, U, U_beta)

            # reproject subspace opt
            if (L_old is not None) and (U_old is not None) and (L_new is not None) and (U_new is not None):
                self.subspace_opt.reproject(U_old=U_old, L_old=L_old, U_new=U_new, L_new=L_new, state=state)

            state["L"] = L_new
            state["U"] = U_new


        if len(history) != 0:
            state['step'] = step + 1 # do not increment if no history (gathering s_ks and y_ks)

    @torch.no_grad
    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):
        U = state.get('U', None)
        if U is None:
            # make a conservative step to avoid issues due to different GD scaling
            return tensor.clip_(-0.1, 0.1)

        # -------------------------------- transforms -------------------------------- #
        L = state['L']
        if "L" in self.children:
            if not self._concat_params: raise RuntimeError("L/U transforms can only be used with concat_params=True")
            L = self.inner_step_tensors("L", [L], clone=True)[0]

        if "U" in self.children:
            if not self._concat_params: raise RuntimeError("L/U transforms can only be used with concat_params=True")
            U = self.inner_step_tensors("U", [U], clone=True)[0]

        # ------------------------------- precondition ------------------------------- #
        g = tensor.view(-1)
        update = opt_lm_adagrad_apply(g, U, L, opt=self.subspace_opt, state=state)
        return update.view_as(tensor)

