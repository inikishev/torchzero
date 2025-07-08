from collections import deque
from operator import itemgetter

import torch

from ...core import Chainable, Module, Transform, Var, apply_transform
from ...utils import NumberList, TensorList, as_tensorlist, unpack_dicts, unpack_states
from ..functional import safe_scaling_
from .lbfgs import _lerp_params_update_


def lsr1_(
    tensors_: TensorList,
    s_history: deque[TensorList],
    y_history: deque[TensorList],
    step: int,
    scale_second: bool,
):
    if len(s_history) == 0:
        # initial step size guess from pytorch
        return safe_scaling_(TensorList(tensors_))

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

    if scale_second and step == 2:
        scale_factor = 1 / TensorList(tensors_).abs().global_sum().clip(min=1)
        scale_factor = scale_factor.clip(min=torch.finfo(tensors_[0].dtype).eps)
        Hx.mul_(scale_factor)

    return Hx


class LSR1(Transform):
    """Limited Memory SR1 algorithm. A line search is recommended.

    .. note::
        L-SR1 provides a better estimate of true hessian, however it is more unstable compared to L-BFGS.

    .. note::
        L-SR1 update rule uses a nested loop, computationally with history size `n` it is similar to L-BFGS with history size `(n^2)/2`. On small problems (ndim <= 2000) BFGS and SR1 may be faster than limited-memory versions.

    .. note::
        directions L-SR1 generates are not guaranteed to be descent directions. This can be alleviated in multiple ways,
        for example using :code:`tz.m.StrongWolfe(plus_minus=True)` line search, or modifying the direction with :code:`tz.m.Cautious` or :code:`tz.m.ScaleByGradCosineSimilarity`.

    Args:
        history_size (int, optional):
            number of past parameter differences and gradient differences to store. Defaults to 10.
        tol (float | None, optional):
            tolerance for minimal parameter difference to avoid instability. Defaults to 1e-10.
        tol_reset (bool, optional):
            If true, whenever gradient difference is less then `tol`, the history will be reset. Defaults to None.
        gtol (float | None, optional):
            tolerance for minimal gradient difference to avoid instability when there is no curvature. Defaults to 1e-10.
        params_beta (float | None, optional):
            if not None, EMA of parameters is used for
            preconditioner update (s_k vector). Defaults to None.
        grads_beta (float | None, optional):
            if not None, EMA of gradients is used for
            preconditioner update (y_k vector). Defaults to None.
        update_freq (int, optional): How often to update L-SR1 history. Defaults to 1.
        scale_second (bool, optional): downscales second update which tends to be large. Defaults to False.
        inner (Chainable | None, optional):
            Optional inner modules applied after updating
            L-SR1 history and before preconditioning. Defaults to None.

    Examples:
        L-SR1 with Strong-Wolfe+- line search

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.LSR1(100),
                tz.m.StrongWolfe(plus_minus=True)
            )
    """
    def __init__(
        self,
        history_size: int = 10,
        tol: float | None = 1e-10,
        tol_reset: bool = False,
        gtol: float | None = 1e-10,
        params_beta: float | None = None,
        grads_beta: float | None = None,
        update_freq: int = 1,
        scale_second: bool = False,
        inner: Chainable | None = None,
    ):
        defaults = dict(
            history_size=history_size, tol=tol, gtol=gtol,
            params_beta=params_beta, grads_beta=grads_beta,
            update_freq=update_freq, scale_second=scale_second,
            tol_reset=tol_reset,
        )
        super().__init__(defaults, uses_grad=False, inner=inner)

        self.global_state['s_history'] = deque(maxlen=history_size)
        self.global_state['y_history'] = deque(maxlen=history_size)

    def reset(self):
        self.state.clear()
        self.global_state['step'] = 0
        self.global_state['s_history'].clear()
        self.global_state['y_history'].clear()

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('prev_l_params', 'prev_l_grad')
        self.global_state.pop('step', None)

    @torch.no_grad
    def update_tensors(self, tensors, params, grads, loss, states, settings):
        params = as_tensorlist(params)
        update = as_tensorlist(tensors)
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        s_history: deque[TensorList] = self.global_state['s_history']
        y_history: deque[TensorList] = self.global_state['y_history']

        setting = settings[0]
        update_freq = itemgetter('update_freq')(setting)

        params_beta, grads_beta = unpack_dicts(settings, 'params_beta', 'grads_beta')
        l_params, l_update = _lerp_params_update_(self, params, update, params_beta, grads_beta)
        prev_l_params, prev_l_grad = unpack_states(states, tensors, 'prev_l_params', 'prev_l_grad', cls=TensorList)

        s = None
        y = None
        if step != 0:
            if step % update_freq == 0:
                s = l_params - prev_l_params
                y = l_update - prev_l_grad

                s_history.append(s)
                y_history.append(y)

        prev_l_params.copy_(l_params)
        prev_l_grad.copy_(l_update)

        # store for apply
        self.global_state['s'] = s
        self.global_state['y'] = y

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        tensors = as_tensorlist(tensors)
        s = self.global_state.pop('s')
        y = self.global_state.pop('y')

        setting = settings[0]
        tol = setting['tol']
        gtol = setting['gtol']
        tol_reset = setting['tol_reset']

        # tolerance on parameter difference to avoid exploding after converging
        if tol is not None:
            if s is not None and s.abs().global_max() <= tol:
                if tol_reset: self.reset()
                return safe_scaling_(TensorList(tensors))

        # tolerance on gradient difference to avoid exploding when there is no curvature
        if tol is not None:
            if y is not None and y.abs().global_max() <= gtol:
                return safe_scaling_(TensorList(tensors))

        # precondition
        dir = lsr1_(
            tensors_=tensors,
            s_history=self.global_state['s_history'],
            y_history=self.global_state['y_history'],
            step=self.global_state.get('step', 1),
            scale_second=setting['scale_second'],
        )

        return dir