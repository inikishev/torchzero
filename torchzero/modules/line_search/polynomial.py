import numpy as np
import torch

from .line_search import LineSearchBase


# polynomial interpolation
# this code is from https://github.com/hjmshi/PyTorch-LBFGS/blob/master/functions/LBFGS.py
# PyTorch-LBFGS: A PyTorch Implementation of L-BFGS
def polyinterp(points, x_min_bound=None, x_max_bound=None, plot=False):
    """
    Gives the minimizer and minimum of the interpolating polynomial over given points
    based on function and derivative information. Defaults to bisection if no critical
    points are valid.

    Based on polyinterp.m Matlab function in minFunc by Mark Schmidt with some slight
    modifications.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 12/6/18.

    Inputs:
        points (nparray): two-dimensional array with each point of form [x f g]
        x_min_bound (float): minimum value that brackets minimum (default: minimum of points)
        x_max_bound (float): maximum value that brackets minimum (default: maximum of points)
        plot (bool): plot interpolating polynomial

    Outputs:
        x_sol (float): minimizer of interpolating polynomial
        F_min (float): minimum of interpolating polynomial

    Note:
      . Set f or g to np.nan if they are unknown

    """
    no_points = points.shape[0]
    order = np.sum(1 - np.isnan(points[:, 1:3]).astype('int')) - 1

    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])

    # compute bounds of interpolation area
    if x_min_bound is None:
        x_min_bound = x_min
    if x_max_bound is None:
        x_max_bound = x_max

    # explicit formula for quadratic interpolation
    if no_points == 2 and order == 2 and plot is False:
        # Solution to quadratic interpolation is given by:
        # a = -(f1 - f2 - g1(x1 - x2))/(x1 - x2)^2
        # x_min = x1 - g1/(2a)
        # if x1 = 0, then is given by:
        # x_min = - (g1*x2^2)/(2(f2 - f1 - g1*x2))

        if points[0, 0] == 0:
            x_sol = -points[0, 2] * points[1, 0] ** 2 / (2 * (points[1, 1] - points[0, 1] - points[0, 2] * points[1, 0]))
        else:
            a = -(points[0, 1] - points[1, 1] - points[0, 2] * (points[0, 0] - points[1, 0])) / (points[0, 0] - points[1, 0]) ** 2
            x_sol = points[0, 0] - points[0, 2]/(2*a)

        x_sol = np.minimum(np.maximum(x_min_bound, x_sol), x_max_bound)

    # explicit formula for cubic interpolation
    elif no_points == 2 and order == 3 and plot is False:
        # Solution to cubic interpolation is given by:
        # d1 = g1 + g2 - 3((f1 - f2)/(x1 - x2))
        # d2 = sqrt(d1^2 - g1*g2)
        # x_min = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2))
        d1 = points[0, 2] + points[1, 2] - 3 * ((points[0, 1] - points[1, 1]) / (points[0, 0] - points[1, 0]))
        d2 = np.sqrt(d1 ** 2 - points[0, 2] * points[1, 2])
        if np.isreal(d2):
            x_sol = points[1, 0] - (points[1, 0] - points[0, 0]) * ((points[1, 2] + d2 - d1) / (points[1, 2] - points[0, 2] + 2 * d2))
            x_sol = np.minimum(np.maximum(x_min_bound, x_sol), x_max_bound)
        else:
            x_sol = (x_max_bound + x_min_bound)/2

    # solve linear system
    else:
        # define linear constraints
        A = np.zeros((0, order + 1))
        b = np.zeros((0, 1))

        # add linear constraints on function values
        for i in range(no_points):
            if not np.isnan(points[i, 1]):
                constraint = np.zeros((1, order + 1))
                for j in range(order, -1, -1):
                    constraint[0, order - j] = points[i, 0] ** j
                A = np.append(A, constraint, 0)
                b = np.append(b, points[i, 1])

        # add linear constraints on gradient values
        for i in range(no_points):
            if not np.isnan(points[i, 2]):
                constraint = np.zeros((1, order + 1))
                for j in range(order):
                    constraint[0, j] = (order - j) * points[i, 0] ** (order - j - 1)
                A = np.append(A, constraint, 0)
                b = np.append(b, points[i, 2])

        # check if system is solvable
        if A.shape[0] != A.shape[1] or np.linalg.matrix_rank(A) != A.shape[0]:
            x_sol = (x_min_bound + x_max_bound)/2
            f_min = np.inf
        else:
            # solve linear system for interpolating polynomial
            coeff = np.linalg.solve(A, b)

            # compute critical points
            dcoeff = np.zeros(order)
            for i in range(len(coeff) - 1):
                dcoeff[i] = coeff[i] * (order - i)

            crit_pts = np.array([x_min_bound, x_max_bound])
            crit_pts = np.append(crit_pts, points[:, 0])

            if not np.isinf(dcoeff).any():
                roots = np.roots(dcoeff)
                crit_pts = np.append(crit_pts, roots)

            # test critical points
            f_min = np.inf
            x_sol = (x_min_bound + x_max_bound) / 2 # defaults to bisection
            for crit_pt in crit_pts:
                if np.isreal(crit_pt) and crit_pt >= x_min_bound and crit_pt <= x_max_bound:
                    F_cp = np.polyval(coeff, crit_pt)
                    if np.isreal(F_cp) and F_cp < f_min:
                        x_sol = np.real(crit_pt)
                        f_min = np.real(F_cp)

            if(plot):
                import matplotlib.pyplot as plt
                plt.figure()
                x = np.arange(x_min_bound, x_max_bound, (x_max_bound - x_min_bound)/10000)
                f = np.polyval(coeff, x)
                plt.plot(x, f)
                plt.plot(x_sol, f_min, 'x')

    return x_sol



# class PolynomialLineSearch(LineSearch):
#     """TODO

#     Line search via polynomial interpolation.

#     Args:
#         init (float, optional): Initial step size. Defaults to 1.0.
#         c1 (float, optional): Acceptance value for weak wolfe condition. Defaults to 1e-4.
#         c2 (float, optional): Acceptance value for strong wolfe condition (set to 0.1 for conjugate gradient). Defaults to 0.9.
#         maxiter (int, optional): Maximum number of line search iterations. Defaults to 25.
#         maxzoom (int, optional): Maximum number of zoom iterations. Defaults to 10.
#         expand (float, optional): Expansion factor (multipler to step size when weak condition not satisfied). Defaults to 2.0.
#         adaptive (bool, optional):
#             when enabled, if line search failed, initial step size is reduced.
#             Otherwise it is reset to initial value. Defaults to True.
#         plus_minus (bool, optional):
#             If enabled and the direction is not descent direction, performs line search in opposite direction. Defaults to False.


#     Examples:
#         Conjugate gradient method with strong wolfe line search. Nocedal, Wright recommend setting c2 to 0.1 for CG.

#         .. code-block:: python

#             opt = tz.Modular(
#                 model.parameters(),
#                 tz.m.PolakRibiere(),
#                 tz.m.StrongWolfe(c2=0.1)
#             )

#         LBFGS strong wolfe line search:

#         .. code-block:: python

#             opt = tz.Modular(
#                 model.parameters(),
#                 tz.m.LBFGS(),
#                 tz.m.StrongWolfe()
#             )

#     """
#     def __init__(
#         self,
#         init: float = 1.0,
#         c1: float = 1e-4,
#         c2: float = 0.9,
#         maxiter: int = 25,
#         maxzoom: int = 10,
#         # a_max: float = 1e10,
#         expand: float = 2.0,
#         adaptive = True,
#         plus_minus = False,
#     ):
#         defaults=dict(init=init,c1=c1,c2=c2,maxiter=maxiter,maxzoom=maxzoom,
#                       expand=expand, adaptive=adaptive, plus_minus=plus_minus)
#         super().__init__(defaults=defaults)

#         self.global_state['initial_scale'] = 1.0
#         self.global_state['beta_scale'] = 1.0

#     @torch.no_grad
#     def search(self, update, var):
#         objective = self.make_objective_with_derivative(var=var)

#         init, c1, c2, maxiter, maxzoom, expand, adaptive, plus_minus = itemgetter(
#             'init', 'c1', 'c2', 'maxiter', 'maxzoom',
#             'expand', 'adaptive', 'plus_minus')(self.settings[var.params[0]])

#         f_0, g_0 = objective(0)

#         step_size,f_a = strong_wolfe(
#             objective,
#             f_0=f_0, g_0=g_0,
#             init=init * self.global_state.setdefault("initial_scale", 1),
#             c1=c1,
#             c2=c2,
#             maxiter=maxiter,
#             maxzoom=maxzoom,
#             expand=expand,
#             plus_minus=plus_minus,
#         )

#         if f_a is not None and (f_a > f_0 or _notfinite(f_a)): step_size = None
#         if step_size is not None and step_size != 0 and not _notfinite(step_size):
#             self.global_state['initial_scale'] = min(1.0, self.global_state['initial_scale'] * math.sqrt(2))
#             return step_size

#         # fallback to backtracking on fail
#         if adaptive: self.global_state['initial_scale'] *= 0.5
#         return 0
