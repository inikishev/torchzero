from collections import deque
from operator import itemgetter
import torch

from ...core import Transform, Chainable, Module, Var, apply_transform
from ...utils import TensorList, as_tensorlist, NumberList
from ...modules.quasi_newton.lbfgs import _adaptive_damping, lbfgs, _lerp_params_update_

class ExpandedLBFGS(Module):
    """L-BFGS but uses differences between more pairs than just consequtive. Window size controls how far away the pairs are allowed to be.
    """
    def __init__(
        self,
        history_size=10,
        window_size:int=3,
        tol: float | None = 1e-10,
        damping: bool = False,
        init_damping=0.9,
        eigval_bounds=(0.5, 50),
        params_beta: float | None = None,
        grads_beta: float | None = None,
        update_freq = 1,
        z_beta: float | None = None,
        tol_reset: bool = False,
        inner: Chainable | None = None,
    ):
        defaults = dict(history_size=history_size, window_size=window_size, tol=tol, damping=damping, init_damping=init_damping, eigval_bounds=eigval_bounds, params_beta=params_beta, grads_beta=grads_beta, update_freq=update_freq, z_beta=z_beta, tol_reset=tol_reset)
        super().__init__(defaults)

        self.global_state['s_history'] = deque(maxlen=history_size)
        self.global_state['y_history'] = deque(maxlen=history_size)
        self.global_state['sy_history'] = deque(maxlen=history_size)
        self.global_state['p_history'] = deque(maxlen=window_size)
        self.global_state['g_history'] = deque(maxlen=window_size)

        if inner is not None:
            self.set_child('inner', inner)

    def reset(self):
        self.state.clear()
        self.global_state['step'] = 0
        self.global_state['s_history'].clear()
        self.global_state['y_history'].clear()
        self.global_state['sy_history'].clear()
        self.global_state['p_history'].clear()
        self.global_state['g_history'].clear()

    @torch.no_grad
    def step(self, var):
        params = as_tensorlist(var.params)
        update = as_tensorlist(var.get_update())
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        # history of s and k
        s_history: deque[TensorList] = self.global_state['s_history']
        y_history: deque[TensorList] = self.global_state['y_history']
        sy_history: deque[torch.Tensor] = self.global_state['sy_history']
        p_history: deque[TensorList] = self.global_state['p_history']
        g_history: deque[TensorList] = self.global_state['g_history']

        tol, damping, init_damping, eigval_bounds, update_freq, z_beta, tol_reset = itemgetter(
            'tol', 'damping', 'init_damping', 'eigval_bounds', 'update_freq', 'z_beta', 'tol_reset')(self.settings[params[0]])
        params_beta, grads_beta = self.get_settings(params, 'params_beta', 'grads_beta')

        l_params, l_update = _lerp_params_update_(self, params, update, params_beta, grads_beta)
        prev_l_params, prev_l_grad = self.get_state(params, 'prev_l_params', 'prev_l_grad', cls=TensorList)

        # 1st step - there are no previous params and grads, lbfgs will do normalized GD step
        if step == 0:
            s = None; y = None; ys = None
        else:
            s = l_params - prev_l_params
            y = l_update - prev_l_grad
            ys = s.dot(y)

            if damping:
                s, y, ys = _adaptive_damping(s, y, ys, init_damping=init_damping, eigval_bounds=eigval_bounds)

        prev_l_params.copy_(l_params)
        prev_l_grad.copy_(l_update)

        # update effective preconditioning state
        if step % update_freq == 0:
            if ys is not None and ys > 1e-10:
                assert s is not None and y is not None
                s_history.append(s)
                y_history.append(y)
                sy_history.append(ys)

            if len(p_history) > 1:
                for p_i, g_i in zip(list(p_history)[:-1], list(g_history)[:-1]):
                    s_i = l_params - p_i
                    y_i = l_update - g_i
                    ys_i = s_i.dot(y_i)

                    if ys_i > 1e-10:
                        if damping:
                            s_i, y_i, ys_i = _adaptive_damping(s_i, y_i, ys_i, init_damping=init_damping, eigval_bounds=eigval_bounds)

                        s_history.append(s_i)
                        y_history.append(y_i)
                        sy_history.append(ys_i)

            p_history.append(l_params.clone())
            g_history.append(l_update.clone())


        # step with inner module before applying preconditioner
        if self.children:
            update = TensorList(apply_transform(self.children['inner'], tensors=update, params=params, grads=var.grad, var=var))

        # tolerance on gradient difference to avoid exploding after converging
        if tol is not None:
            if y is not None and y.abs().global_max() <= tol:
                var.update = update # may have been updated by inner module, probably makes sense to use it here?
                if tol_reset: self.reset()
                return var

        # lerp initial H^-1 @ q guess
        z_ema = None
        if z_beta is not None:
            z_ema = self.get_state(var.params, 'z_ema', cls=TensorList)

        # precondition
        dir = lbfgs(
            tensors_=as_tensorlist(update),
            s_history=s_history,
            y_history=y_history,
            sy_history=sy_history,
            y=y,
            sy=ys,
            z_beta = z_beta,
            z_ema = z_ema,
            step=step
        )

        var.update = dir

        return var

