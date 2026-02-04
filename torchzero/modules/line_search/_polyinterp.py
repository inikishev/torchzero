import numpy as np
import torch

from .line_search import LineSearchBase
from ...utils import tofloat

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
        value = d1 ** 2 - points[0, 2] * points[1, 2]
        if value > 0:
            d2 = np.sqrt(value)
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
                if np.isreal(crit_pt):
                    if not np.isrealobj(crit_pt): crit_pt = crit_pt.real
                    if crit_pt >= x_min_bound and crit_pt <= x_max_bound:
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


# polynomial interpolation
# this code is based on https://github.com/hjmshi/PyTorch-LBFGS/blob/master/functions/LBFGS.py
# PyTorch-LBFGS: A PyTorch Implementation of L-BFGS
# this one is modified where instead of clipping the solution by bounds, it tries a lower degree polynomial
# all the way to bisection
def _within_bounds(x, lb, ub):
    if lb is not None and x < lb: return False
    if ub is not None and x > ub: return False
    return True

def _quad_interp(points):
    assert points.shape[0] == 2, points.shape
    if points[0, 0] == 0:
        denom = 2 * (points[1, 1] - points[0, 1] - points[0, 2] * points[1, 0])
        if abs(denom) > 1e-32:
            return -points[0, 2] * points[1, 0] ** 2 / denom
    else:
        denom = (points[0, 0] - points[1, 0]) ** 2
        if denom > 1e-32:
            a = -(points[0, 1] - points[1, 1] - points[0, 2] * (points[0, 0] - points[1, 0])) / denom
            if a > 1e-32:
                return points[0, 0] - points[0, 2]/(2*a)
    return None

def _cubic_interp(points, lb, ub):
    assert points.shape[0] == 2, points.shape
    denom = points[0, 0] - points[1, 0]
    if abs(denom) > 1e-32:
        d1 = points[0, 2] + points[1, 2] - 3 * ((points[0, 1] - points[1, 1]) / denom)
        value = d1 ** 2 - points[0, 2] * points[1, 2]
        if value > 0:
            d2 = np.sqrt(value)
            denom = points[1, 2] - points[0, 2] + 2 * d2
            if abs(denom) > 1e-32:
                x_sol = points[1, 0] - (points[1, 0] - points[0, 0]) * ((points[1, 2] + d2 - d1) / denom)
                if _within_bounds(x_sol, lb, ub): return x_sol

    # try quadratic interpolations
    x_sol = _quad_interp(points)
    if x_sol is not None and _within_bounds(x_sol, lb, ub): return x_sol

    return None

def _poly_interp(points, lb, ub):
    no_points = points.shape[0]
    assert no_points > 2, points.shape
    order = np.sum(1 - np.isnan(points[:, 1:3]).astype('int')) - 1

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
        return None

    # solve linear system for interpolating polynomial
    coeff = np.linalg.solve(A, b)

    # compute critical points
    dcoeff = np.zeros(order)
    for i in range(len(coeff) - 1):
        dcoeff[i] = coeff[i] * (order - i)

    lower = np.min(points[:, 0]) if lb is None else lb
    upper = np.max(points[:, 0]) if ub is None else ub

    crit_pts = np.array([lower, upper])
    crit_pts = np.append(crit_pts, points[:, 0])

    if not np.isinf(dcoeff).any():
        roots = np.roots(dcoeff)
        crit_pts = np.append(crit_pts, roots)

    # test critical points
    f_min = np.inf
    x_sol = None
    for crit_pt in crit_pts:
        if np.isreal(crit_pt):
            if not np.isrealobj(crit_pt): crit_pt = crit_pt.real
            if _within_bounds(crit_pt, lb, ub):
                F_cp = np.polyval(coeff, crit_pt)
                if np.isreal(F_cp) and F_cp < f_min:
                    x_sol = np.real(crit_pt)
                    f_min = np.real(F_cp)

    return x_sol

def polyinterp2(points, lb, ub, unbounded: bool = False):
    no_points = points.shape[0]
    if no_points <= 1:
        return (lb + ub)/2

    order = np.sum(1 - np.isnan(points[:, 1:3]).astype('int')) - 1

    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])

    # compute bounds of interpolation area
    if not unbounded:
        if lb is None:
            lb = x_min
        if ub is None:
            ub = x_max

    if no_points == 2 and order == 2:
        x_sol = _quad_interp(points)
        if x_sol is not None and _within_bounds(x_sol, lb, ub): return x_sol
        return (lb + ub)/2

    if no_points == 2 and order == 3:
        x_sol = _cubic_interp(points, lb, ub) # includes fallback on _quad_interp
        if x_sol is not None and _within_bounds(x_sol, lb, ub): return x_sol
        return (lb + ub)/2

    if no_points <= 2: # order < 2
        return (lb + ub)/2

    if no_points == 3:
        for p in (points[:2], points[1:], points[::2]):
            x_sol = _cubic_interp(p, lb, ub)
            if x_sol is not None and _within_bounds(x_sol, lb, ub): return x_sol

    if lb is not None: lb = tofloat(lb)
    if ub is not None: ub = tofloat(ub)
    x_sol = _poly_interp(points, lb, ub)
    if x_sol is not None and _within_bounds(x_sol, lb, ub): return x_sol
    return polyinterp2(points[1:], lb, ub)