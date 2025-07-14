"""this needs to be reworked maybe but it also works"""
import math
import warnings
from operator import itemgetter

import torch
from torch.optim.lbfgs import _cubic_interpolate

from .line_search import LineSearchBase
from ...utils import totensor


def _zoom(f,
          a_l, a_h,
          f_l, g_l,
          f_h, g_h,
          f_0, g_0,
          c1, c2,
          maxzoom):

    for i in range(maxzoom):
        a_j = _cubic_interpolate(
            *(totensor(i) for i in (a_l, f_l, g_l, a_h, f_h, g_h))

        )

        # if interpolation fails or produces endpoint, bisect
        delta = abs(a_h - a_l)
        if a_j is None or a_j == a_l or a_j == a_h:
            a_j = a_l + 0.5 * delta


        f_j, g_j = f(a_j)

        # check armijo
        armijo = f_j <= f_0 + c1 * a_j * g_0

        # check strong wolfe
        wolfe = abs(g_j) <= c2 * abs(g_0)


        # minimum between alpha_low and alpha_j
        if not armijo or f_j >= f_l:
            a_h = a_j
            f_h = f_j
            g_h = g_j
        else:
            # alpha_j satisfies armijo
            if wolfe:
                return a_j, f_j

            # minimum between alpha_j and alpha_high
            if g_j * (a_h - a_l) >= 0:
                # between alpha_low and alpha_j
                # a_h = a_l
                # f_h = f_l
                # g_h = g_l
                a_h = a_j
                f_h = f_j
                g_h = g_j

            # is this messing it up?
            else:
                a_l = a_j
                f_l = f_j
                g_l = g_j




        # check if interval too small
        delta = abs(a_h - a_l)
        if delta <= 1e-9 or delta <= 1e-6 * max(abs(a_l), abs(a_h)):
            l_satisfies_wolfe = (f_l <= f_0 + c1 * a_l * g_0) and (abs(g_l) <= c2 * abs(g_0))
            h_satisfies_wolfe = (f_h <= f_0 + c1 * a_h * g_0) and (abs(g_h) <= c2 * abs(g_0))

            if l_satisfies_wolfe and h_satisfies_wolfe: return a_l if f_l <= f_h else a_h, f_h
            if l_satisfies_wolfe: return a_l, f_l
            if h_satisfies_wolfe: return a_h, f_h
            if f_l <= f_0 + c1 * a_l * g_0: return a_l, f_l
            return None,None

        if a_j is None or a_j == a_l or a_j == a_h:
            a_j = a_l + 0.5 * delta


    return None,None


def strong_wolfe(
    f,
    f_0,
    g_0,
    init: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    maxiter: int = 25,
    maxzoom: int = 15,
    # a_max: float = 1e30,
    expand: float = 2.0,  # Factor to increase alpha in bracketing
    plus_minus: bool = False,
) -> tuple[float,float] | tuple[None,None]:
    a_prev = 0.0

    if g_0 == 0: return None,None
    if g_0 > 0:
        # if direction is not a descent direction, perform line search in opposite direction
        if plus_minus:
            def inverted_objective(alpha):
                l, g = f(-alpha)
                return l, -g
            a, v = strong_wolfe(
                inverted_objective,
                init=init,
                f_0=f_0,
                g_0=-g_0,
                c1=c1,
                c2=c2,
                maxiter=maxiter,
                # a_max=a_max,
                expand=expand,
                plus_minus=False,
            )
            if a is not None and v is not None: return -a, v
        return None, None

    f_prev = f_0
    g_prev = g_0
    a_cur = init

    # bracket
    for i in range(maxiter):

        f_cur, g_cur = f(a_cur)

        # check armijo
        armijo_violated = f_cur > f_0 + c1 * a_cur * g_0
        func_increased = f_cur >= f_prev and i > 0

        if armijo_violated or func_increased:
            return _zoom(f,
                         a_prev, a_cur,
                         f_prev, g_prev,
                         f_cur, g_cur,
                         f_0, g_0,
                         c1, c2,
                         maxzoom=maxzoom,
                         )



        # check strong wolfe
        if abs(g_cur) <= c2 * abs(g_0):
            return a_cur, f_cur

        # minimum is bracketed
        if g_cur >= 0:
            return _zoom(f,
                        #alpha_curr, alpha_prev,
                        a_prev, a_cur,
                        #phi_curr, phi_prime_curr,
                        f_prev, g_prev,
                        f_cur, g_cur,
                        f_0, g_0,
                        c1, c2,
                        maxzoom=maxzoom,)

        # otherwise continue bracketing
        a_next = a_cur * expand

        # update previous point and continue loop with increased step size
        a_prev = a_cur
        f_prev = f_cur
        g_prev = g_cur
        a_cur = a_next


    # max iters reached
    return None, None

def _notfinite(x):
    if isinstance(x, torch.Tensor): return not torch.isfinite(x).all()
    return not math.isfinite(x)

class StrongWolfe(LineSearchBase):
    """Cubic interpolation line search satisfying Strong Wolfe condition.

    Args:
        init (float, optional): Initial step size. Defaults to 1.0.
        c1 (float, optional): Acceptance value for weak wolfe condition. Defaults to 1e-4.
        c2 (float, optional): Acceptance value for strong wolfe condition (set to 0.1 for conjugate gradient). Defaults to 0.9.
        maxiter (int, optional): Maximum number of line search iterations. Defaults to 25.
        maxzoom (int, optional): Maximum number of zoom iterations. Defaults to 10.
        expand (float, optional): Expansion factor (multipler to step size when weak condition not satisfied). Defaults to 2.0.
        use_prev (bool, optional):
            if True, previous step size is used as the initial step size on the next step.
        adaptive (bool, optional):
            when enabled, if line search failed, initial step size is reduced.
            Otherwise it is reset to initial value. Defaults to True.
        plus_minus (bool, optional):
            If enabled and the direction is not descent direction, performs line search in opposite direction. Defaults to False.


    Examples:
        Conjugate gradient method with strong wolfe line search. Nocedal, Wright recommend setting c2 to 0.1 for CG.

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.PolakRibiere(),
                tz.m.StrongWolfe(c2=0.1)
            )

        LBFGS strong wolfe line search:

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.LBFGS(),
                tz.m.StrongWolfe()
            )

    """
    def __init__(
        self,
        init: float = 1.0,
        c1: float = 1e-4,
        c2: float = 0.9,
        maxiter: int = 25,
        maxzoom: int = 10,
        # a_max: float = 1e10,
        expand: float = 2.0,
        use_prev: bool = False,
        adaptive = True,
        plus_minus = False,
    ):
        defaults=dict(init=init,c1=c1,c2=c2,maxiter=maxiter,maxzoom=maxzoom,
                      expand=expand, adaptive=adaptive, plus_minus=plus_minus,use_prev=use_prev)
        super().__init__(defaults=defaults)

        self.global_state['initial_scale'] = 1.0
        self.global_state['beta_scale'] = 1.0

    @torch.no_grad
    def search(self, update, var):
        objective = self.make_objective_with_derivative(var=var)

        init, c1, c2, maxiter, maxzoom, expand, adaptive, plus_minus, use_prev = itemgetter(
            'init', 'c1', 'c2', 'maxiter', 'maxzoom',
            'expand', 'adaptive', 'plus_minus', 'use_prev')(self.settings[var.params[0]])

        f_0, g_0 = objective(0)
        if use_prev: init = self.global_state.get('prev_alpha', init)

        step_size,f_a = strong_wolfe(
            objective,
            f_0=f_0, g_0=g_0,
            init=init * self.global_state.setdefault("initial_scale", 1),
            c1=c1,
            c2=c2,
            maxiter=maxiter,
            maxzoom=maxzoom,
            expand=expand,
            plus_minus=plus_minus,
        )

        if f_a is not None and (f_a > f_0 or _notfinite(f_a)): step_size = None
        if step_size is not None and step_size != 0 and not _notfinite(step_size):
            self.global_state['initial_scale'] = min(1.0, self.global_state['initial_scale'] * math.sqrt(2))
            self.global_state['prev_alpha'] = step_size
            return step_size

        if adaptive: self.global_state['initial_scale'] *= 0.5
        return 0
