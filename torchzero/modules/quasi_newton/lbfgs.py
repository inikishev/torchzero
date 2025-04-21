from collections import deque

import torch

from ...core import Transform
from ...utils import TensorList, as_tensorlist


class LBFGS(Transform):
    def __init__(self, history_size=10):
        defaults = dict(history_size = history_size)
        super().__init__(defaults, uses_grad=False)

        self.global_state['s_history'] = deque(maxlen=history_size)
        self.global_state['y_history'] = deque(maxlen=history_size)
        self.global_state['sy_history'] = deque(maxlen=history_size)
        self.global_state['step'] = 0

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        params = vars.params
        grad = as_tensorlist(target) # fr brevity
        prev_params, prev_grad = self.get_state('prev_params', 'prev_grad', params=params, cls=TensorList, init=[params, grad])

        # history of s and k
        s_history: deque[TensorList] = self.global_state['s_history']
        y_history: deque[TensorList] = self.global_state['y_history']
        sy_history: deque[torch.Tensor] = self.global_state['sy_history']

        # 1st step
        if self.global_state['step'] == 0:
            # dir = params.grad.sign() # may work fine

            # initial step size guess taken from pytorch L-BFGS
            dir = grad.mul_(min(1.0, 1.0 / grad.abs().global_sum())) # pyright: ignore[reportArgumentType]

        else:
            s_k = params - prev_params
            y_k = grad - prev_grad
            sy_k = s_k.dot(y_k)

            # only add pair if curvature is positive
            if sy_k > 0:
                s_history.append(s_k)
                y_history.append(y_k)
                sy_history.append(sy_k)

            #else:
                # print(f'negative curvature: {sy_k}')

            prev_params.copy_(params)
            prev_grad.copy_(grad)

            # lbfgs part
            # 1st loop
            alpha_list = []
            q = grad.clone()
            z = None
            for s_i, y_i, sy_i in zip(reversed(s_history), reversed(y_history), reversed(sy_history)):
                p_i = 1 / sy_i # this is also denoted as ρ (rho)
                alpha = p_i * s_i.dot(q)
                alpha_list.append(alpha)
                q.sub_(y_i, alpha=alpha) # pyright: ignore[reportArgumentType]

            # calculate z
            # s.y/y.y is also this weird y-looking symbol I couldn't find
            # z is it times q
            # actually H0 = (s.y/y.y) * I, and z = H0 @ q

            z = q * (sy_k / (y_k.dot(y_k)))

            assert z is not None

            # 2nd loop
            for s_i, y_i, sy_i, alpha_i in zip(s_history, y_history, sy_history, reversed(alpha_list)):
                p_i = 1 / sy_i
                beta_i = p_i * y_i.dot(z)
                z.add_(s_i, alpha = alpha_i - beta_i)

            dir = z

        self.global_state['step'] += 1
        return dir