r"""
Line searches.
"""

import typing as T

from ...core import OptimizerModule
from ..meta.chain import Chain
from ..misc import Normalize
from .grid_ls import (ArangeLS, BacktrackingLS, GridLS, LinspaceLS,
                      MultiplicativeLS)
from .quad_interp import QuadraticInterpolation2Point
from .directional_newton import DirectionalNewton3Points, DirectionalNewton
from .scipy_minimize_scalar import ScipyMinimizeScalarLS

LineSearches = T.Literal['backtracking', 'brent', 'brent-exact', 'brent-norm', 'multiplicative', 'newton', 'newton-grad'] | OptimizerModule

def get_line_search(name:str | OptimizerModule):
    if isinstance(name, str):
        name = name.strip().lower()
        if name == 'backtracking': return BacktrackingLS()
        if name == 'multiplicative': return MultiplicativeLS()
        if name == 'brent': return ScipyMinimizeScalarLS(maxiter=8)
        if name == 'brent-exact': return ScipyMinimizeScalarLS()
        if name == 'brent-norm': return [Normalize(), ScipyMinimizeScalarLS(maxiter=16)]
        if name == 'newton': return DirectionalNewton3Points(1)
        if name == 'newton-grad': return DirectionalNewton(1)
        raise ValueError(f"Unknown line search method: {name}")
    return name