import math
from bisect import insort

import numpy as np
from numpy.polynomial import Polynomial


# we have a list of points in ascending order of their `y` value
class Point:
    __slots__ = ("x", "y", "d")
    def __init__(self, x, y, d):
        self.x = x
        self.y = y
        self.d = d

    def __lt__(self, other):
        return self.y < other.y

def _get_dpoint(points: list[Point]):
    """returns lowest point with derivative and list of other points"""
    for i,p in enumerate(points):
        if p.d is not None:
            cpoints = points.copy()
            del cpoints[i]
            return p, cpoints
    return None, points

# -------------------------------- quadratic2 -------------------------------- #
def _fitmin_quadratic2(x1, y1, d1, x2, y2):

    a = (y2 - y1 - d1*(x2 - x1)) / (x2 - x1)**2
    if a <= 0: return None

    b = d1 - 2*a*x1
    # c = y_1 - d_1*x_1 + a*x_1**2

    return -b / (2*a)

def quadratic2(points:list[Point]):
    pd, points = _get_dpoint(points)
    if pd is None: return None
    if len(points) == 0: return None

    pn = points[0]
    return _fitmin_quadratic2(pd.x, pd.y, pd.d, pn.x, pn.y)

# -------------------------------- quadratic3 -------------------------------- #
def _fitmin_quadratic3(x1, y1, x2, y2, x3, y3):
    quad = Polynomial.fit([x1,x2,x3], [y1,y2,y3], deg=2)
    a,b,c = quad.coef
    if a <= 0: return None
    return -b / (2*a)

def quadratic3(points:list[Point]):
    if len(points) < 3: return None

    p1,p2,p3 = points[:3]
    return _fitmin_quadratic3(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y)

# ---------------------------------- cubic3 ---------------------------------- #
def _minimize_polynomial(poly: Polynomial):
    roots = poly.deriv().roots()
    vals = poly(roots)
    argmin = np.argmin(vals)
    return roots[argmin], vals[argmin]


def _fitmin_cubic3(x1,y1,x2,y2,x3,y3,x4,d4):
    """x4 is allowed to be equal to x1"""

    A = np.array([
        [x1**3, x1**2, x1, 1],
        [x2**3, x2**2, x2, 1],
        [x3**3, x3**2, x3, 1],
        [3*x4**2, 2*x4, 1, 0]
    ])

    B = np.array([y1, y2, y3, d4])

    try:
        coeffs = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return None

    cubic = Polynomial(coeffs)
    x_min, y_min = _minimize_polynomial(cubic)
    if y_min < min(y1,y2,y3): return x_min
    return None

def cubic3(points: list[Point]):
    pd, points = _get_dpoint(points)
    if pd is None: return None
    if len(points) < 2: return None
    p1, p2 = points[:2]
    return _fitmin_cubic3(pd.x, pd.y, p1.x, p1.y, p2.x, p2.y, pd.x, pd.d)

# ---------------------------------- cubic4 ---------------------------------- #
def _fitmin_cubic4(x1, y1, x2, y2, x3, y3, x4, y4):
    cubic = Polynomial.fit([x1,x2,x3,x4], [y1,y2,y3,y4], deg=3)
    x_min, y_min = _minimize_polynomial(cubic)
    if y_min < min(y1,y2,y3,y4): return x_min
    return None

def cubic4(points:list[Point]):
    if len(points) < 4: return None

    p1,p2,p3,p4 = points[:4]
    return _fitmin_cubic4(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y)

# ---------------------------------- linear3 --------------------------------- #
def _linear_intersection(x1,y1,s1,x2,y2,s2):
    if s1 == 0 or s2 == 0 or s1 == s2: return None
    return (y1 - s1*x1 - y2 + s2*x2) / (s2 - s1)

def _fitmin_linear3(x1, y1, d1, x2, y2, x3, y3):
    # we have that
    # s2 = (y2 - y3) / (x2 - x3) # slope origin in x2 y2
    # f1(x) = y1 + d1 * (x - x1)
    # f2(x) = y2 + s2 * (x - x2)
    # y1 + d1 * (x - x1) = y2 + s2 * (x - x2)
    # y1 + d1 x - d1 x1 - y2 - s2 x + s2 x2 = 0
    # s2 x - d1 x = y1 - d1 x1 - y2 + s2 x2
    # x = (y1 - d1 x1 - y2 + s2 x2) / (s2 - d1)

    if x2 < x1 < x3 or x3 < x1 < x2: # point with derivative in between
        return None

    if d1 > 0:
        if x2 > x1 or x3 > x1: return None  # intersection is above to the right
        if x2 > x3: x2,y2,x3,y3 = x3,y3,x2,y2
    if d1 < 0:
        if x2 < x1 or x3 < x1: return None  # intersection is above to the left
        if x2 < x3: x2,y2,x3,y3 = x3,y3,x2,y2

    s2 = (y2 - y3) / (x2 - x3)
    return _linear_intersection(x1,y1,d1,x2,y2,s2)

def linear3(points:list[Point]):
    pd, points = _get_dpoint(points)
    if pd is None: return None
    if len(points) < 2: return None
    p1, p2 = points[:2]
    return _fitmin_linear3(pd.x, pd.y, pd.d, p1.x, p1.y, p2.x, p2.y)

# ---------------------------------- linear4 --------------------------------- #
def _fitmin_linear4(x1, y1, x2, y2, x3, y3, x4, y4):
    # sort by x
    points = ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
    points = sorted(points, key=lambda x: x[0])

    (x1,y1), (x2,y2), (x3,y3), (x4,y4) = points
    s1 = (y1 - y2) / (x1 - x2)
    s3 = (y3 - y4) / (x3 - x4)

    return _linear_intersection(x1,y1,s1,x3,y3,s3)

def linear4(points:list[Point]):
    if len(points) < 4: return None
    p1,p2,p3,p4 = points[:4]
    return _fitmin_linear4(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y)