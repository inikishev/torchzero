from collections import deque
from operator import itemgetter

import torch

from ...core import Chainable, Module, Transform, Var, apply_transform
from ...utils import NumberList, TensorList, as_tensorlist

from .lbfgs import _lerp_params_update_

def lsr1_(
    tensors_: TensorList,
    s_history: deque[TensorList],
    y_history: deque[TensorList],
    step: int,
    scale_second: bool,
):
    if step == 0 or not s_history:
        # initial step size guess from pytorch
        scale_factor = 1 / TensorList(tensors_).abs().global_sum().clip(min=1)
        scale_factor = scale_factor.clip(min=torch.finfo(tensors_[0].dtype).eps)
        return tensors_.mul_(scale_factor)

    m = len(s_history)

    w_list: list[TensorList] = []
    ww_list: list = [None for _ in range(m)]
    wy_list: list = [None for _ in range(m)]

    # 1st loop - all w_k = s_k - H_k_prev y_k
    for k in range(m):
        s_k = s_history[k]
        y_k = y_history[k]

        H_k = y_k.clone()
        for j in range(k):
            w_j = w_list[j]
            y_j = y_history[j]

            wy = wy_list[j]
            if wy is None: wy = wy_list[j] = w_j.dot(y_j)

            ww = ww_list[j]
            if ww is None: ww = ww_list[j] = w_j.dot(w_j)

            if wy == 0: continue

            H_k.add_(w_j, alpha=w_j.dot(y_k) / wy) # pyright:ignore[reportArgumentType]

        w_k = s_k - H_k
        w_list.append(w_k)

    Hx = tensors_.clone()
    for k in range(m):
        w_k = w_list[k]
        y_k = y_history[k]
        wy = wy_list[k]
        ww = ww_list[k]

        if wy is None: wy = w_k.dot(y_k) # this happens when m = 1 so inner loop doesn't run
        if ww is None: ww = w_k.dot(w_k)

        if wy == 0: continue

        Hx.add_(w_k, alpha=w_k.dot(tensors_) / wy) # pyright:ignore[reportArgumentType]

    if scale_second and step == 1:
        scale_factor = 1 / TensorList(tensors_).abs().global_sum().clip(min=1)
        scale_factor = scale_factor.clip(min=torch.finfo(tensors_[0].dtype).eps)
        Hx.mul_(scale_factor)

    return Hx


class LSR1(Module):
    """Limited Memory SR1 (L-SR1)
    Args:
        history_size (int, optional): Number of past parameter differences (s)
            and gradient differences (y) to store. Defaults to 10.
        skip_R_val (float, optional): Tolerance R for the SR1 update skip condition
            |w_k^T y_k| >= R * ||w_k|| * ||y_k||. Defaults to 1e-8.
            Updates where this condition is not met are skipped during history accumulation
            and matrix-vector products.
        params_beta (float | None, optional): If not None, EMA of parameters is used for
            preconditioner update (s_k vector). Defaults to None.
        grads_beta (float | None, optional): If not None, EMA of gradients is used for
            preconditioner update (y_k vector). Defaults to None.
        update_freq (int, optional): How often to update L-SR1 history. Defaults to 1.
        conv_tol (float | None, optional): Tolerance for y_k norm. If max abs value of y_k
            is below this, the preconditioning step might be skipped, assuming convergence.
            Defaults to 1e-10.
        inner (Chainable | None, optional): Optional inner modules applied after updating
            L-SR1 history and before preconditioning. Defaults to None.
    """
    def __init__(
        self,
        history_size: int = 10,
        tol: float = 1e-8,
        params_beta: float | None = None,
        grads_beta: float | None = None,
        update_freq: int = 1,
        scale_second: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = dict(
            history_size=history_size, tol=tol,
            params_beta=params_beta, grads_beta=grads_beta,
            update_freq=update_freq, scale_second=scale_second
        )
        super().__init__(defaults)

        self.global_state['s_history'] = deque(maxlen=history_size)
        self.global_state['y_history'] = deque(maxlen=history_size)

        if inner is not None:
            self.set_child('inner', inner)

    def reset(self):
        self.state.clear()
        self.global_state['step'] = 0
        self.global_state['s_history'].clear()
        self.global_state['y_history'].clear()


    @torch.no_grad
    def step(self, var: Var):
        params = as_tensorlist(var.params)
        update = as_tensorlist(var.get_update())
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        s_history: deque[TensorList] = self.global_state['s_history']
        y_history: deque[TensorList] = self.global_state['y_history']

        settings = self.settings[params[0]]
        tol, update_freq, scale_second = itemgetter('tol', 'update_freq', 'scale_second')(settings)

        params_beta, grads_beta_ = self.get_settings(params, 'params_beta', 'grads_beta') # type: ignore
        l_params, l_update = _lerp_params_update_(self, params, update, params_beta, grads_beta_)

        prev_l_params, prev_l_grad = self.get_state(params, 'prev_l_params', 'prev_l_grad', cls=TensorList)

        y_k = None
        if step != 0:
            if step % update_freq == 0:
                s_k = l_params - prev_l_params
                y_k = l_update - prev_l_grad

                s_history.append(s_k)
                y_history.append(y_k)

        prev_l_params.copy_(l_params)
        prev_l_grad.copy_(l_update)

        if 'inner' in self.children:
            update = TensorList(apply_transform(self.children['inner'], tensors=update, params=params, grads=var.grad, var=var))

        # tolerance on gradient difference to avoid exploding after converging
        if tol is not None:
            if y_k is not None and y_k.abs().global_max() <= tol:
                var.update = update
                return var

        dir = lsr1_(
            tensors_=update,
            s_history=s_history,
            y_history=y_history,
            step=step,
            scale_second=scale_second,
        )

        var.update = dir

        return var