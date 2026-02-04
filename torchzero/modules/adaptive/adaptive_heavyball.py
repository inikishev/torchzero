import torch
from ...core import  TensorTransform
from ...utils import TensorList, unpack_dicts, unpack_states


def adaptive_heavy_ball(f, f_star, f_prev, g: TensorList, g_prev: TensorList, p: TensorList, p_prev: TensorList):
    if f - f_star <= torch.finfo(p[0].dtype).tiny * 2: return g

    g_g = g.dot(g)
    g_gp = g.dot(g_prev)
    num = -(f - f_star) * g.dot(g_prev)
    denom = (f_prev - f_star) * g_g + (f - f_star) * g_gp
    m = num/denom

    h = 2*(f - f_star) / g_g
    return (1 + m) * h * g - m*(p-p_prev)


class AdaptiveHeavyBall(TensorTransform):
    """Adaptive heavy ball from https://hal.science/hal-04832983v1/file/OJMO_2024__5__A7_0.pdf.

    Suitable for quadratic objectives with known f* (loss at minimum).

    note:
        The step size is determined by the algorithm, so learning rate modules shouldn't be used.

    Args:
        f_star (int, optional):
            (estimated) minimal possible value of the objective function (lowest possible loss). Defaults to 0.
    """
    def __init__(self, f_star: float = 0):
        defaults = dict(f_star=f_star)
        super().__init__(defaults, uses_loss=True)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        assert loss is not None
        tensors = TensorList(tensors)
        f_star = settings[0]['f_star']

        f_prev = self.global_state.get('f_prev', None)
        p_prev, g_prev = unpack_states(states, tensors, 'p_prev', 'g_prev', init=[params,tensors], cls=TensorList)

        # -------------------------------- first step -------------------------------- #
        if f_prev is None:
            self.global_state['f_prev'] = loss
            h = 2*(loss - f_star) / tensors.dot(tensors)
            return h * tensors

        # ------------------------------- further steps ------------------------------ #
        update = adaptive_heavy_ball(
            f=loss, f_star=f_star, f_prev=f_prev, g=tensors, g_prev=g_prev, p=TensorList(params), p_prev=p_prev)

        # --------------------------- store previous values -------------------------- #
        self.global_state['f_prev'] = loss
        p_prev.copy_(params)
        g_prev.copy_(tensors)

        return update