from collections import deque
from functools import partial
from operator import itemgetter
from typing import Literal

import torch

from ...core import Chainable, Module, Transform, Var, apply_transform
from ...utils import NumberList, TensorList, as_tensorlist
from .lbfgs import _adaptive_damping, lbfgs


@torch.no_grad
def _store_sk_yk_after_step_hook(optimizer, var: Var, prev_params: TensorList, prev_grad: TensorList, damping, init_damping, eigval_bounds, s_history: deque[TensorList], y_history: deque[TensorList], sy_history: deque[torch.Tensor]):
    assert var.closure is not None
    with torch.enable_grad(): var.closure()
    grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in var.params]
    s_k = var.params - prev_params
    y_k = grad - prev_grad
    ys_k = s_k.dot(y_k)

    if damping:
        s_k, y_k, ys_k = _adaptive_damping(s_k, y_k, ys_k, init_damping=init_damping, eigval_bounds=eigval_bounds)

    if ys_k > 1e-10:
        s_history.append(s_k)
        y_history.append(y_k)
        sy_history.append(ys_k)



class OnlineLBFGS(Module):
    """Online L-BFGS.
    Parameter and gradient differences are sampled from the same mini-batch by performing an extra forward and backward pass.
    However I did a bunch of experiments and the online part doesn't seem to help. Normal L-BFGS is usually still
    better because it performs twice as many steps, and it is reasonably stable with normalization or grafting.

    Args:
        history_size (int, optional): number of past parameter differences and gradient differences to store. Defaults to 10.
        sample_grads (str, optional):
            - "before" - samples current mini-batch gradient at previous and current parameters, calculates y_k
            and adds it to history before stepping.
            - "after" - samples current mini-batch gradient at parameters before stepping and after updating parameters.
                s_k and y_k are added after parameter update, therefore they are delayed by 1 step.

            In practice both modes behave very similarly. Defaults to 'before'.
        tol (float | None, optional):
            tolerance for minimal gradient difference to avoid instability after converging to minima. Defaults to 1e-10.
        damping (bool, optional):
            whether to use adaptive damping. Learning rate might need to be lowered with this enabled. Defaults to False.
        init_damping (float, optional):
            initial damping for adaptive dampening. Defaults to 0.9.
        eigval_bounds (tuple, optional):
            eigenvalue bounds for adaptive dampening. Defaults to (0.5, 50).
        params_beta (float | None, optional):
            if not None, EMA of parameters is used for preconditioner update. Defaults to None.
        grads_beta (float | None, optional):
            if not None, EMA of gradients is used for preconditioner update. Defaults to None.
        update_freq (int, optional):
            how often to update L-BFGS history. Defaults to 1.
        z_beta (float | None, optional):
            optional EMA for initial H^-1 @ q. Acts as a kind of momentum but is prone to get stuck. Defaults to None.
        inner (Chainable | None, optional):
            optional inner modules applied after updating L-BFGS history and before preconditioning. Defaults to None.
    """
    def __init__(
        self,
        history_size=10,
        sample_grads: Literal['before', 'after'] = 'before',
        tol: float | None = 1e-10,
        damping: bool = False,
        init_damping=0.9,
        eigval_bounds=(0.5, 50),
        z_beta: float | None = None,
        inner: Chainable | None = None,
    ):
        defaults = dict(history_size=history_size, tol=tol, damping=damping, init_damping=init_damping, eigval_bounds=eigval_bounds, sample_grads=sample_grads, z_beta=z_beta)
        super().__init__(defaults)

        self.global_state['s_history'] = deque(maxlen=history_size)
        self.global_state['y_history'] = deque(maxlen=history_size)
        self.global_state['sy_history'] = deque(maxlen=history_size)

        if inner is not None:
            self.set_child('inner', inner)

    def reset(self):
        """Resets the internal state of the L-SR1 module."""
        # super().reset() # Clears self.state (per-parameter) if any, and "step"
        # Re-initialize L-SR1 specific global state
        self.state.clear()
        self.global_state['step'] = 0
        self.global_state['s_history'].clear()
        self.global_state['y_history'].clear()
        self.global_state['sy_history'].clear()

    @torch.no_grad
    def step(self, var):
        assert var.closure is not None

        params = as_tensorlist(var.params)
        update = as_tensorlist(var.get_update())
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        # history of s and k
        s_history: deque[TensorList] = self.global_state['s_history']
        y_history: deque[TensorList] = self.global_state['y_history']
        sy_history: deque[torch.Tensor] = self.global_state['sy_history']

        tol, damping, init_damping, eigval_bounds, sample_grads, z_beta = itemgetter(
            'tol', 'damping', 'init_damping', 'eigval_bounds', 'sample_grads', 'z_beta')(self.settings[params[0]])

        # sample gradient at previous params with current mini-batch
        if sample_grads == 'before':
            prev_params = self.get_state(params, 'prev_params', cls=TensorList)
            if step == 0:
                s_k = None; y_k = None; ys_k = None
            else:
                s_k = params - prev_params

                current_params = params.clone()
                params.set_(prev_params)
                with torch.enable_grad(): var.closure()
                y_k = update - params.grad
                ys_k = s_k.dot(y_k)
                params.set_(current_params)

                if damping:
                    s_k, y_k, ys_k = _adaptive_damping(s_k, y_k, ys_k, init_damping=init_damping, eigval_bounds=eigval_bounds)

                if ys_k > 1e-10:
                    s_history.append(s_k)
                    y_history.append(y_k)
                    sy_history.append(ys_k)

            prev_params.copy_(params)

        # use previous s_k, y_k pair, samples gradient at current batch before and after updating parameters
        elif sample_grads == 'after':
            if len(s_history) == 0:
                s_k = None; y_k = None; ys_k = None
            else:
                s_k = s_history[-1]
                y_k = y_history[-1]
                ys_k = s_k.dot(y_k)

            # this will run after params are updated by Modular after running all future modules
            var.post_step_hooks.append(
                partial(
                    _store_sk_yk_after_step_hook,
                    prev_params=params.clone(),
                    prev_grad=update.clone(),
                    damping=damping,
                    init_damping=init_damping,
                    eigval_bounds=eigval_bounds,
                    s_history=s_history,
                    y_history=y_history,
                    sy_history=sy_history,
                ))

        else:
            raise ValueError(sample_grads)

        # step with inner module before applying preconditioner
        if self.children:
            update = TensorList(apply_transform(self.children['inner'], tensors=update, params=params, grads=var.grad, var=var))

        # tolerance on gradient difference to avoid exploding after converging
        if tol is not None:
            if y_k is not None and y_k.abs().global_max() <= tol:
                var.update = update # may have been updated by inner module, probably makes sense to use it here?
                return var

        # lerp initial H^-1 @ q guess
        z_ema = None
        if z_beta is not None:
            z_ema = self.get_state(params, 'z_ema', cls=TensorList)

        # precondition
        dir = lbfgs(
            tensors_=as_tensorlist(update),
            s_history=s_history,
            y_history=y_history,
            sy_history=sy_history,
            y_k=y_k,
            ys_k=ys_k,
            z_beta = z_beta,
            z_ema = z_ema,
            step=step
        )

        var.update = dir

        return var

