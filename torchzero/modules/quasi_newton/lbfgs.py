from collections import deque
from operator import itemgetter
import torch

from ...core import Transform
from ...utils import TensorList, as_tensorlist, NumberList



def _update_lbfgs_history_(
    params: TensorList,
    grad: TensorList,
    prev_params_: TensorList,
    prev_grad_: TensorList,
    s_history: deque[TensorList],
    y_history: deque[TensorList],
    sy_history: deque[torch.Tensor],
    damping = False,
    init_damping = 0.99,
    eigval_bounds = (0.01, 1.5)
):

    s_k = params - prev_params_
    y_k = grad - prev_grad_
    ys_k = s_k.dot(y_k)

    if damping:
        # adaptive damping Al-Baali, M.: Quasi-Wolfe conditions for quasi-Newton methods for large-scale optimization. In: 40th Workshop on Large Scale Nonlinear Optimization, Erice, Italy, June 22–July 1 (2004)
        sigma_l, sigma_h = eigval_bounds
        u = ys_k / s_k.dot(s_k)
        if u <= sigma_l < 1: tau = min((1-sigma_l)/(1-u), init_damping)
        elif u >= sigma_h > 1: tau = min((sigma_h-1)/(u-1), init_damping)
        else: tau = init_damping
        y_k = tau * y_k + (1-tau) * s_k
        ys_k = s_k.dot(y_k)

    # only add pair if curvature is positive
    if ys_k > 1e-10:
        s_history.append(s_k)
        y_history.append(y_k)
        sy_history.append(ys_k)

    #else:
        # print(f'negative curvature: {sy_k}')

    prev_params_.copy_(params)
    prev_grad_.copy_(grad)

    return y_k, ys_k

def lbfgs(
    tensors_: TensorList,
    s_history: deque[TensorList],
    y_history: deque[TensorList],
    sy_history: deque[torch.Tensor],
    y_k: TensorList,
    ys_k: torch.Tensor,
    step: int,
):
    if step == 0 or len(s_history) == 0:
        # dir = params.grad.sign() # may work fine

        # initial step size guess taken from pytorch L-BFGS
        return tensors_.mul_(min(1.0, 1.0 / tensors_.abs().global_sum())) # pyright: ignore[reportArgumentType]

    else:
        # 1st loop
        alpha_list = []
        q = tensors_.clone()
        z = None
        for s_i, y_i, ys_i in zip(reversed(s_history), reversed(y_history), reversed(sy_history)):
            p_i = 1 / ys_i # this is also denoted as ρ (rho)
            alpha = p_i * s_i.dot(q)
            alpha_list.append(alpha)
            q.sub_(y_i, alpha=alpha) # pyright: ignore[reportArgumentType]

        # calculate z
        # s.y/y.y is also this weird y-looking symbol I couldn't find
        # z is it times q
        # actually H0 = (s.y/y.y) * I, and z = H0 @ q
        z = q * (ys_k / (y_k.dot(y_k)))

        assert z is not None

        # 2nd loop
        for s_i, y_i, ys_i, alpha_i in zip(s_history, y_history, sy_history, reversed(alpha_list)):
            p_i = 1 / ys_i
            beta_i = p_i * y_i.dot(z)
            z.add_(s_i, alpha = alpha_i - beta_i)

        return z


class LBFGS(Transform):
    def __init__(self, history_size=10, tol:float|None=1e-10, damping:bool = False, init_damping = 0.9, eigval_bounds = (0.5, 50)):
        defaults = dict(history_size=history_size, tol=tol, damping=damping, init_damping=init_damping, eigval_bounds=eigval_bounds)
        super().__init__(defaults, uses_grad=False)

        self.global_state['s_history'] = deque(maxlen=history_size)
        self.global_state['y_history'] = deque(maxlen=history_size)
        self.global_state['sy_history'] = deque(maxlen=history_size)
        self.global_state['step'] = 0

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        prev_params, prev_grad = self.get_state('prev_params', 'prev_grad', params=params, cls=TensorList, init=[params, target])
        params = as_tensorlist(params)
        target = as_tensorlist(target)

        # history of s and k
        s_history: deque[TensorList] = self.global_state['s_history']
        y_history: deque[TensorList] = self.global_state['y_history']
        sy_history: deque[torch.Tensor] = self.global_state['sy_history']

        tol, damping, init_damping, eigval_bounds = itemgetter(
            'tol', 'damping', 'init_damping', 'eigval_bounds')(self.settings[params[0]])

        y_k, ys_k = _update_lbfgs_history_(
            params=params,
            grad=target,
            prev_params_=prev_params,
            prev_grad_=prev_grad,
            s_history=s_history,
            y_history=y_history,
            sy_history=sy_history,
            damping=damping,
            init_damping=init_damping,
            eigval_bounds=eigval_bounds,
        )

        if tol is not None and self.global_state['step'] != 0: # it will be 0 on 1st step
            if y_k.abs().global_max() <= tol: return target

        dir = lbfgs(
            tensors_=target,
            s_history=s_history,
            y_history=y_history,
            sy_history=sy_history,
            y_k=y_k,
            ys_k=ys_k,
            step=self.global_state['step']
        )

        self.global_state['step'] += 1
        return dir

