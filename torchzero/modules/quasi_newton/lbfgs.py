from collections import deque
from collections.abc import Sequence
from typing import overload

import torch

from ...core import Chainable, TensorTransform
from ...utils import TensorList, as_tensorlist, unpack_states
from ...linalg.linear_operator import LinearOperator
from ..opt_utils import initial_step_size
from .damping import DampingStrategyType, apply_damping


@torch.no_grad
def _make_M(S:torch.Tensor, Y:torch.Tensor, B_0:torch.Tensor):
    m,n = S.size()

    M = torch.zeros((2 * m, 2 * m), device=S.device, dtype=S.dtype)

    # top-left is B S^T S
    M[:m, :m] = B_0 * S @ S.mT

    # anti-diagonal is L^T and L
    L = (S @ Y.mT).tril_(-1)

    M[m:, :m] = L.mT
    M[:m, m:] = L

    # bottom-right
    D_diag = (S * Y).sum(1).neg()
    M[m:, m:] = D_diag.diag_embed()

    return M


@torch.no_grad
def lbfgs_Bx(x: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, sy_history, M=None):
    """L-BFGS hessian-vector product based on compact representation,
    returns (Bx, M), where M is an internal matrix that depends on S and Y so it can be reused."""
    m = len(S)
    if m == 0: return x.clone()

    # initial scaling
    y = Y[-1]
    sy = sy_history[-1]
    yy = y.dot(y)
    B_0 = yy / sy
    Bx = x * B_0

    Psi = torch.zeros(2 * m, device=x.device, dtype=x.dtype)
    Psi[:m] = B_0 * S@x
    Psi[m:] = Y@x

    if M is None: M = _make_M(S, Y, B_0)

    # solve Mu = p
    u, info = torch.linalg.solve_ex(M, Psi) # pylint:disable=not-callable
    if info != 0:
        return Bx

    # Bx
    u_S = u[:m]
    u_Y = u[m:]
    SuS = (S * u_S.unsqueeze(-1)).sum(0)
    YuY = (Y * u_Y.unsqueeze(-1)).sum(0)
    return Bx - (B_0 * SuS + YuY), M


@overload
def lbfgs_Hx(
    x: torch.Tensor,
    s_history: Sequence[torch.Tensor] | torch.Tensor,
    y_history: Sequence[torch.Tensor] | torch.Tensor,
    sy_history: Sequence[torch.Tensor] | torch.Tensor,
) -> torch.Tensor: ...
@overload
def lbfgs_Hx(
    x: TensorList,
    s_history: Sequence[TensorList],
    y_history: Sequence[TensorList],
    sy_history: Sequence[torch.Tensor] | torch.Tensor,
) -> TensorList: ...
def lbfgs_Hx(
    x,
    s_history: Sequence | torch.Tensor,
    y_history: Sequence | torch.Tensor,
    sy_history: Sequence[torch.Tensor] | torch.Tensor,
):
    """L-BFGS inverse-hessian-vector product, works with tensors and TensorLists"""
    x = x.clone()
    if len(s_history) == 0: return x

    # 1st loop
    alpha_list = []
    for s_i, y_i, sy_i in zip(reversed(s_history), reversed(y_history), reversed(sy_history)):
        p_i = 1 / sy_i
        alpha = p_i * s_i.dot(x)
        alpha_list.append(alpha)
        x.sub_(y_i, alpha=alpha)

    # scaled initial hessian inverse
    # H_0 = (s.y/y.y) * I, and z = H_0 @ q
    sy = sy_history[-1]
    y = y_history[-1]
    Hx = x * (sy / y.dot(y))

    # 2nd loop
    for s_i, y_i, sy_i, alpha_i in zip(s_history, y_history, sy_history, reversed(alpha_list)):
        p_i = 1 / sy_i
        beta_i = p_i * y_i.dot(Hx)
        Hx.add_(s_i, alpha = alpha_i - beta_i)

    return Hx


class LBFGSLinearOperator(LinearOperator):
    def __init__(self, s_history: Sequence[torch.Tensor] | torch.Tensor, y_history: Sequence[torch.Tensor] | torch.Tensor, sy_history: Sequence[torch.Tensor] | torch.Tensor):
        super().__init__()
        if len(s_history) == 0:
            self.S = self.Y = self.yy = None
        else:
            self.S = s_history
            self.Y = y_history

        self.sy_history = sy_history
        self.M = None

    def _get_S(self):
        if self.S is None: return None
        if not isinstance(self.S, torch.Tensor):
            self.S = torch.stack(tuple(self.S))
        return self.S

    def _get_Y(self):
        if self.Y is None: return None
        if not isinstance(self.Y, torch.Tensor):
            self.Y = torch.stack(tuple(self.Y))
        return self.Y

    def solve(self, b):
        S = self._get_S(); Y = self._get_Y()
        if S is None or Y is None: return b.clone()
        return lbfgs_Hx(b, S, Y, self.sy_history)

    def matvec(self, x):
        S = self._get_S(); Y = self._get_Y()
        if S is None or Y is None: return x.clone()
        Bx, self.M = lbfgs_Bx(x, S, Y, self.sy_history, M=self.M)
        return Bx

    def size(self):
        if self.S is None: raise RuntimeError()
        n = len(self.S[0])
        return (n, n)


class LBFGS(TensorTransform):
    """Limited-memory BFGS algorithm. A line search or trust region is recommended.

    Args:
        history_size (int, optional):
            number of past parameter differences and gradient differences to store. Defaults to 10.
        ptol (float | None, optional):
            skips updating the history if maximum absolute value of
            parameter difference is less than this value. Defaults to 1e-10.
        ptol_restart (bool, optional):
            If true, whenever parameter difference is less then ``ptol``,
            L-BFGS state will be reset. Defaults to None.
        gtol (float | None, optional):
            skips updating the history if if maximum absolute value of
            gradient difference is less than this value. Defaults to 1e-10.
        ptol_restart (bool, optional):
            If true, whenever gradient difference is less then ``gtol``,
            L-BFGS state will be reset. Defaults to None.
        sy_tol (float | None, optional):
            history will not be updated whenever s⋅y is less than this value (negative s⋅y means negative curvature)
        scale_first (bool, optional):
            makes first step, when hessian approximation is not available,
            small to reduce number of line search iterations. Defaults to True.
        update_freq (int, optional):
            how often to update L-BFGS history. Larger values may be better for stochastic optimization. Defaults to 1.
        damping (DampingStrategyType, optional):
            damping to use, can be "powell" or "double". Defaults to None.
        inner (Chainable | None, optional):
            optional inner modules applied after updating L-BFGS history and before preconditioning. Defaults to None.

    ## Examples:

    L-BFGS with line search
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.LBFGS(100),
        tz.m.Backtracking()
    )
    ```

    L-BFGS with trust region
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.TrustCG(tz.m.LBFGS())
    )
    ```
    """
    def __init__(
        self,
        history_size=10,
        ptol: float | None = 1e-32,
        ptol_restart: bool = False,
        gtol: float | None = 1e-32,
        gtol_restart: bool = False,
        sy_tol: float = 1e-32,
        scale_first:bool=True,
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
            sy_tol=sy_tol,
            damping = damping,
        )
        super().__init__(defaults, inner=inner, update_freq=update_freq)

        self.global_state['s_history'] = deque(maxlen=history_size)
        self.global_state['y_history'] = deque(maxlen=history_size)
        self.global_state['sy_history'] = deque(maxlen=history_size)

    def _reset_self(self):
        self.state.clear()
        self.global_state['step'] = 0
        self.global_state['s_history'].clear()
        self.global_state['y_history'].clear()
        self.global_state['sy_history'].clear()

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
        s_history: deque[TensorList] = self.global_state['s_history']
        y_history: deque[TensorList] = self.global_state['y_history']
        sy_history: deque[torch.Tensor] = self.global_state['sy_history']

        ptol = self.defaults['ptol']
        gtol = self.defaults['gtol']
        ptol_restart = self.defaults['ptol_restart']
        gtol_restart = self.defaults['gtol_restart']
        sy_tol = self.defaults['sy_tol']
        damping = self.defaults['damping']

        p_prev, g_prev = unpack_states(states, tensors, 'p_prev', 'g_prev', cls=TensorList)

        # 1st step - there are no previous params and grads, lbfgs will do normalized SGD step
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
                if ptol_restart:
                    self._reset_self()
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
        if sy is not None and sy > sy_tol:
            assert s is not None and y is not None and sy is not None

            s_history.append(s)
            y_history.append(y)
            sy_history.append(sy)

    def get_H(self, objective=...):
        s_history = [tl.to_vec() for tl in self.global_state['s_history']]
        y_history = [tl.to_vec() for tl in self.global_state['y_history']]
        sy_history = self.global_state['sy_history']
        return LBFGSLinearOperator(s_history, y_history, sy_history)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        scale_first = self.defaults['scale_first']

        tensors = as_tensorlist(tensors)

        s_history = self.global_state['s_history']
        y_history = self.global_state['y_history']
        sy_history = self.global_state['sy_history']

        # precondition
        dir = lbfgs_Hx(
            x=tensors,
            s_history=s_history,
            y_history=y_history,
            sy_history=sy_history,
        )

        # scale 1st step
        if scale_first and self.global_state.get('step', 1) == 1:
            dir *= initial_step_size(dir, eps=1e-7)

        return dir