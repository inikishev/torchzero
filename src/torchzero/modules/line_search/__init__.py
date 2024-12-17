r"""
Line searches.
"""

import typing as T

from ...core import OptimizerModule
from ..meta import Chain
from ..operators import Normalize
from .grid_ls import (ArangeLS, BacktrackingLS, GridLS, LinspaceLS,
                      MultiplicativeLS)
from .quad_interp import QuadraticInterpolation2Point
from .quadratic_ls import MinimizeQuadratic3PointsLS, MinimizeQuadraticLS
from .scipy_minimize_scalar import ScipyMinimizeScalarLS

LineSearches = T.Literal['backtracking', 'brent', 'brent-exact', 'brent-norm', 'multiplicative'] | OptimizerModule

def get_line_search(name:str | OptimizerModule):
    if isinstance(name, str):
        name = name.strip().lower()
        if name == 'backtracking': return BacktrackingLS()
        if name == 'multiplicative': return BacktrackingLS()
        elif name == 'brent': return ScipyMinimizeScalarLS(maxiter=8)
        elif name == 'brent-exact': return ScipyMinimizeScalarLS()
        elif name == 'brent-norm': return Chain([Normalize(), ScipyMinimizeScalarLS(maxiter=16)])
        else: raise ValueError(f"Unknown line search method: {name}")
    else:
        return name