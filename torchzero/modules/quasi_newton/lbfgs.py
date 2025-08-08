from collections import deque
from collections.abc import Sequence
from operator import itemgetter
from typing import overload

import torch

from ...core import Chainable, Module, Transform, Var, apply_transform
from ...utils import NumberList, TensorList, as_tensorlist, unpack_dicts, unpack_states
from ...utils.linalg.linear_operator import LinearOperator
from ..functional import initial_step_size

@torch.no_grad
def _make_M(S:torch.Tensor, Y:torch.Tensor, B_0:torch.Tensor):
    m,n = S.size()

    M = torch.zeros((2 * m, 2 * m), device=S.device, dtype=S.dtype)

    # top-left S^T * B^0 * S = B * S^T * S
    M[:m, :m] = B_0 * S @ S.mT

    # anti-diagonal is L
    L = (S @ Y.mT).tril_(-1)

    M[m:, :m] = L.mT
    M[:m, m:] = L

    # bottom-right block: -D (diagonal matrix)
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
    yy = y @ y
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
    tensors: torch.Tensor,
    s_history: Sequence[torch.Tensor] | torch.Tensor,
    y_history: Sequence[torch.Tensor] | torch.Tensor,
    sy_history: Sequence[torch.Tensor],
) -> torch.Tensor: ...
@overload
def lbfgs_Hx(
    tensors: TensorList,
    s_history: Sequence[TensorList],
    y_history: Sequence[TensorList],
    sy_history: Sequence[torch.Tensor],
) -> TensorList: ...
def lbfgs_Hx(
    tensors,
    s_history: Sequence | torch.Tensor,
    y_history: Sequence | torch.Tensor,
    sy_history: Sequence[torch.Tensor],
):
    """works with tensors and TensorLists"""
    q = tensors.clone()
    if len(s_history) == 0: return q

    alpha_list = []
    for s_i, y_i, sy_i in zip(reversed(s_history), reversed(y_history), reversed(sy_history)):
        p_i = 1 / sy_i
        alpha = p_i * s_i.dot(q)
        alpha_list.append(alpha)
        q.sub_(y_i, alpha=alpha)

    # scaled initial hessian inverse
    # H_0 = (s.y/y.y) * I, and z = H_0 @ q
    sy = sy_history[-1]
    y = y_history[-1]
    z = q * (sy / y.dot(y))

    # 2nd loop
    for s_i, y_i, sy_i, alpha_i in zip(s_history, y_history, sy_history, reversed(alpha_list)):
        p_i = 1 / sy_i
        beta_i = p_i * y_i.dot(z)
        z.add_(s_i, alpha = alpha_i - beta_i)

    return z


class LBFGSLinearOperator(LinearOperator):
    def __init__(self, s_history: Sequence[torch.Tensor], y_history: Sequence[torch.Tensor], sy_history: Sequence[torch.Tensor]):
        super().__init__()
        if len(s_history) == 0:
            self.S = self.Y = self.yy = None
        else:
            self.S = torch.stack(tuple(s_history))
            self.Y = torch.stack(tuple(y_history))
            self.yy = self.Y[-1].dot(self.Y[-1])

        self.sy_history = sy_history

        self.M = None

    def solve(self, b):
        if self.S is None: return b.clone()
        assert self.Y is not None
        return lbfgs_Hx(b, self.S, self.Y, self.sy_history)

    def matvec(self, x):
        if self.S is None: return x.clone()
        assert self.Y is not None
        Bx, self.M = lbfgs_Bx(x, self.S, self.Y, self.sy_history, M=self.M)
        return Bx



class LBFGS(Transform):
    """Limited-memory BFGS algorithm. A line search is recommended, although L-BFGS may be reasonably stable without it.

    Args:
        history_size (int, optional):
            number of past parameter differences and gradient differences to store. Defaults to 10.
        tol (float | None, optional):
            tolerance for minimal parameter difference to avoid instability. Defaults to 1e-10.
        tol_reset (bool, optional):
            If true, whenever gradient difference is less then `tol`, the history will be reset. Defaults to None.
        gtol (float | None, optional):
            tolerance for minimal gradient difference to avoid instability when there is no curvature. Defaults to 1e-10.
        update_freq (int, optional):
            how often to update L-BFGS history. Defaults to 1.
        inner (Chainable | None, optional):
            optional inner modules applied after updating L-BFGS history and before preconditioning. Defaults to None.

    Examples:
        L-BFGS with strong-wolfe line search

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.LBFGS(100),
                tz.m.StrongWolfe()
            )

        L-BFGS preconditioning applied to momentum (may be unstable!)

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.LBFGS(inner=tz.m.EMA(0.9)),
                tz.m.LR(1e-2)
            )
    """
    def __init__(
        self,
        history_size=10,
        ptol: float | None = 1e-10,
        ptol_reset: bool = False,
        gtol: float | None = 1e-10,
        gtol_reset: bool = False,
        sy_tol: float = 1e-10,
        scale_first:bool=True,
        update_freq = 1,
        inner: Chainable | None = None,
    ):
        defaults = dict(
            history_size=history_size,
            scale_first=scale_first,
            ptol=ptol,
            gtol=gtol,
            ptol_reset=ptol_reset,
            gtol_reset=gtol_reset,
            sy_tol=sy_tol,
        )
        super().__init__(defaults, uses_grad=False, inner=inner, update_freq=update_freq)

        self.global_state['s_history'] = deque(maxlen=history_size)
        self.global_state['y_history'] = deque(maxlen=history_size)
        self.global_state['sy_history'] = deque(maxlen=history_size)

    def reset(self):
        self.state.clear()
        self.global_state['step'] = 0
        self.global_state['s_history'].clear()
        self.global_state['y_history'].clear()
        self.global_state['sy_history'].clear()
        for c in self.children.values(): c.reset()

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('p_prev', 'g_prev')
        self.global_state.pop('step', None)

    @torch.no_grad
    def update_tensors(self, tensors, params, grads, loss, states, settings):
        p = as_tensorlist(params)
        g = as_tensorlist(tensors)
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        # history of s and k
        s_history: deque[TensorList] = self.global_state['s_history']
        y_history: deque[TensorList] = self.global_state['y_history']
        sy_history: deque[torch.Tensor] = self.global_state['sy_history']

        setting = settings[0]
        ptol = setting['ptol']
        gtol = setting['gtol']
        ptol_reset = setting['ptol_reset']
        gtol_reset = setting['gtol_reset']
        sy_tol = setting['sy_tol']

        p_prev, g_prev = unpack_states(states, tensors, 'p_prev', 'g_prev', cls=TensorList)

        # 1st step - there are no previous params and grads, lbfgs will do normalized SGD step
        if step == 0:
            s = None; y = None; sy = None
        else:
            s = p - p_prev
            y = g - g_prev
            sy = s.dot(y)
            # damping to be added here

        below_tol = False
        # tolerance on parameter difference to avoid exploding after converging
        if ptol is not None:
            if s is not None and s.abs().global_max() <= ptol:
                if ptol_reset: self.reset()
                sy = None
                below_tol = True

        # tolerance on gradient difference to avoid exploding when there is no curvature
        if gtol is not None:
            if y is not None and y.abs().global_max() <= gtol:
                if gtol_reset: self.reset()
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

    def get_H(self, var):
        s_history = [tl.to_vec() for tl in self.global_state['s_history']]
        y_history = [tl.to_vec() for tl in self.global_state['y_history']]
        sy_history = self.global_state['sy_history']
        return LBFGSLinearOperator(s_history, y_history, sy_history)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        setting = settings[0]
        scale_first = setting['scale_first']

        tensors = as_tensorlist(tensors)

        s_history = self.global_state['s_history']
        y_history = self.global_state['y_history']
        sy_history = self.global_state['sy_history']

        # precondition
        dir = lbfgs_Hx(
            tensors=tensors,
            s_history=s_history,
            y_history=y_history,
            sy_history=sy_history,
        )

        # scale 1st step
        if scale_first and self.global_state.get('step', 1) == 1:
            dir *= initial_step_size(dir, eps=1e-7)

        return dir