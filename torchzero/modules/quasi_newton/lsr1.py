from collections import deque
from collections.abc import Sequence
from operator import itemgetter

import torch

from ...core import Chainable, Module, TensorTransform, Objective, step
from ...utils import NumberList, TensorList, as_tensorlist, generic_finfo_tiny, unpack_states, vec_to_tensors_
from ...linalg.linear_operator import LinearOperator
from ..opt_utils import initial_step_size
from .damping import DampingStrategyType, apply_damping


def lsr1_Hx(x, s_history: Sequence, y_history: Sequence,):
    m = len(s_history)
    if m == 0: return x.clone()
    eps = generic_finfo_tiny(x) * 2

    w_list = []
    wy_list: list = [None for _ in range(m)]

    # # 1st loop - all w_k = s_k - H_k_prev y_k
    for k in range(m):
        s_k = s_history[k]
        y_k = y_history[k]

        Hx = y_k.clone()
        for j in range(k):
            w_j = w_list[j]
            y_j = y_history[j]

            wy = wy_list[j]
            if wy is None: wy = wy_list[j] = w_j.dot(y_j)
            if wy.abs() < eps: continue

            alpha = w_j.dot(y_k) / wy
            Hx.add_(w_j, alpha=alpha)

        w_k = s_k - Hx
        w_list.append(w_k)

    Hx = x.clone()

    # second loop
    for k in range(m):
        w_k = w_list[k]
        y_k = y_history[k]
        wy = wy_list[k]

        if wy is None: wy = w_k.dot(y_k) # this happens when m = 1 so inner loop doesn't run
        if wy.abs() < eps: continue

        alpha = w_k.dot(x) / wy
        Hx.add_(w_k, alpha=alpha)

    return Hx

def lsr1_Bx(x, s_history: Sequence, y_history: Sequence,):
    return lsr1_Hx(x, s_history=y_history, y_history=s_history)

class LSR1LinearOperator(LinearOperator):
    def __init__(self, s_history: Sequence[torch.Tensor], y_history: Sequence[torch.Tensor]):
        super().__init__()
        self.s_history = s_history
        self.y_history = y_history

    def solve(self, b):
        return lsr1_Hx(x=b, s_history=self.s_history, y_history=self.y_history)

    def matvec(self, x):
        return lsr1_Bx(x=x, s_history=self.s_history, y_history=self.y_history)

    def size(self):
        if len(self.s_history) == 0: raise RuntimeError()
        n = len(self.s_history[0])
        return (n, n)


class LSR1(TensorTransform):
    """Limited-memory SR1 algorithm. A line search or trust region is recommended.

    Args:
        history_size (int, optional):
            number of past parameter differences and gradient differences to store. Defaults to 10.
        ptol (float | None, optional):
            skips updating the history if maximum absolute value of
            parameter difference is less than this value. Defaults to None.
        ptol_restart (bool, optional):
            If true, whenever parameter difference is less then ``ptol``,
            L-SR1 state will be reset. Defaults to None.
        gtol (float | None, optional):
            skips updating the history if if maximum absolute value of
            gradient difference is less than this value. Defaults to None.
        ptol_restart (bool, optional):
            If true, whenever gradient difference is less then ``gtol``,
            L-SR1 state will be reset. Defaults to None.
        scale_first (bool, optional):
            makes first step, when hessian approximation is not available,
            small to reduce number of line search iterations. Defaults to False.
        update_freq (int, optional):
            how often to update L-SR1 history. Larger values may be better for stochastic optimization. Defaults to 1.
        damping (DampingStrategyType, optional):
            damping to use, can be "powell" or "double". Defaults to None.
        compact (bool, optional):
            if True, uses a compact representation verstion of L-SR1. It is much faster computationally, but less stable.
        inner (Chainable | None, optional):
            optional inner modules applied after updating L-SR1 history and before preconditioning. Defaults to None.

    ## Examples:

    L-SR1 with line search
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.SR1(),
        tz.m.StrongWolfe(c2=0.1, fallback=True)
    )
    ```

    L-SR1 with trust region
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.TrustCG(tz.m.LSR1())
    )
    ```
    """
    def __init__(
        self,
        history_size=10,
        ptol: float | None = None,
        ptol_restart: bool = False,
        gtol: float | None = None,
        gtol_restart: bool = False,
        scale_first:bool=False,
        update_freq = 1,
        damping: DampingStrategyType = None,
        inner: Chainable | None = None,
    ):
        defaults = dict(
            history_size=history_size,
            scale_first=scale_first,
            ptol=ptol,
            gtol=gtol,
            ptol_restart=ptol_restart,
            gtol_restart=gtol_restart,
            damping = damping,
        )
        super().__init__(defaults, inner=inner, update_freq=update_freq)

        self.global_state['s_history'] = deque(maxlen=history_size)
        self.global_state['y_history'] = deque(maxlen=history_size)

    def _reset_self(self):
        self.state.clear()
        self.global_state['step'] = 0
        self.global_state['s_history'].clear()
        self.global_state['y_history'].clear()

    def reset(self):
        self._reset_self()
        for c in self.children.values(): c.reset()

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('p_prev', 'g_prev')
        self.global_state.pop('step', None)

    @torch.no_grad
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):
        p = as_tensorlist(params)
        g = as_tensorlist(tensors)
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        # history of s and k
        s_history: deque = self.global_state['s_history']
        y_history: deque = self.global_state['y_history']

        ptol = self.defaults['ptol']
        gtol = self.defaults['gtol']
        ptol_restart = self.defaults['ptol_restart']
        gtol_restart = self.defaults['gtol_restart']
        damping = self.defaults['damping']

        p_prev, g_prev = unpack_states(states, tensors, 'p_prev', 'g_prev', cls=TensorList)

        # 1st step - there are no previous params and grads, lsr1 will do normalized SGD step
        if step == 0:
            s = None; y = None; sy = None
        else:
            s = p - p_prev
            y = g - g_prev

            if damping is not None:
                s, y = apply_damping(damping, s=s, y=y, g=g, H=self.get_H())

            sy = s.dot(y)
            # damping to be added here

        below_tol = False
        # tolerance on parameter difference to avoid exploding after converging
        if ptol is not None:
            if s is not None and s.abs().global_max() <= ptol:
                if ptol_restart: self._reset_self()
                sy = None
                below_tol = True

        # tolerance on gradient difference to avoid exploding when there is no curvature
        if gtol is not None:
            if y is not None and y.abs().global_max() <= gtol:
                if gtol_restart: self._reset_self()
                sy = None
                below_tol = True

        # store previous params and grads
        if not below_tol:
            p_prev.copy_(p)
            g_prev.copy_(g)

        # update effective preconditioning state
        if sy is not None:
            assert s is not None and y is not None and sy is not None

            s_history.append(s)
            y_history.append(y)

    def get_H(self, objective=...):
        s_history = [tl.to_vec() for tl in self.global_state['s_history']]
        y_history = [tl.to_vec() for tl in self.global_state['y_history']]
        return LSR1LinearOperator(s_history, y_history)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        scale_first = self.defaults['scale_first']

        tensors = as_tensorlist(tensors)

        s_history = self.global_state['s_history']
        y_history = self.global_state['y_history']

        # precondition
        dir = lsr1_Hx(
            x=tensors,
            s_history=s_history,
            y_history=y_history,
        )

        # scale 1st step
        if scale_first and self.global_state.get('step', 1) == 1:
            dir *= initial_step_size(dir, eps=1e-7)

        return dir
