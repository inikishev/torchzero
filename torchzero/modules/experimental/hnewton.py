from collections import deque

import torch

from ...core import TensorwiseTransform


def eigh_solve(H: torch.Tensor, g: torch.Tensor):
    try:
        L, Q = torch.linalg.eigh(H) # pylint:disable=not-callable
        return Q @ ((Q.mH @ g) / L)
    except torch.linalg.LinAlgError:
        return None


class HNewton(TensorwiseTransform):
    """This treats gradient differences as Hvps with vectors being parameter differences, using past gradients that are close to each other. Basically this is another limited memory quasi newton method to test.

    .. warning::
        Experimental.

    """
    def __init__(self, history_size: int, window_size: int, reg: float=0, tol: float = 1e-8, concat_params:bool=True, inner=None):
        defaults = dict(history_size=history_size, window_size=window_size, reg=reg, tol=tol)
        super().__init__(defaults, uses_grad=False, concat_params=concat_params, inner=inner)

    def update_tensor(self, tensor, param, grad, loss, state, setting):

        history_size = setting['history_size']

        if 'param_history' not in state:
            state['param_history'] = deque(maxlen=history_size)
            state['grad_history'] = deque(maxlen=history_size)

        param_history: deque = state['param_history']
        grad_history: deque = state['grad_history']
        param_history.append(param.ravel())
        grad_history.append(tensor.ravel())

    def apply_tensor(self, tensor, param, grad, loss, state, setting):
        window_size = setting['window_size']
        reg = setting['reg']
        tol = setting['tol']

        param_history: deque = state['param_history']
        grad_history: deque = state['grad_history']
        g = tensor.ravel()

        n = len(param_history)
        s_list = []
        y_list = []

        for i in range(n):
            for j in range(i):
                if i - j <= window_size:
                    p_i, g_i = param_history[i], grad_history[i]
                    p_j, g_j = param_history[j], grad_history[j]
                    s = p_i - p_j # vec in hvp
                    y = g_i - g_j # hvp
                    if s.dot(y) > tol:
                        s_list.append(s)
                        y_list.append(y)

        if len(s_list) < 1:
            scale = (1 / tensor.abs().sum()).clip(min=torch.finfo(tensor.dtype).eps, max=1)
            tensor.mul_(scale)
            return tensor

        S = torch.stack(s_list, 1)
        Y = torch.stack(y_list, 1)

        B = S.T @ Y
        if reg != 0: B.add_(torch.eye(B.size(0), device=B.device, dtype=B.dtype).mul_(reg))
        g_proj = g @ S

        newton_proj, info = torch.linalg.solve_ex(B, g_proj) # pylint:disable=not-callable
        if info != 0:
            newton_proj = -torch.linalg.lstsq(B, g_proj).solution # pylint:disable=not-callable
        newton = S @ newton_proj
        return newton.view_as(tensor)


        # scale = (1 / tensor.abs().sum()).clip(min=torch.finfo(tensor.dtype).eps, max=1)
        # tensor.mul_(scale)
        # return tensor