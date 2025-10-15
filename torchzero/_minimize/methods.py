"""WIP API"""
import itertools
import time
from collections import deque
from collections.abc import Callable, Sequence, Mapping, Iterable
from typing import Any, NamedTuple, cast, overload

import numpy as np
import torch

from .. import m
from ..core import Module, Optimizer
from ..utils import tofloat


def _get_method_from_str(method: str) -> list[Module]:
    method = ''.join(c for c in method.lower().strip() if c.isalnum())

    if method == "bfgs":
        return [m.RestartOnStuck(m.BFGS()), m.Backtracking()]

    if method == "lbfgs":
        return [m.LBFGS(100), m.Backtracking()]

    if method == "newton":
        return [m.Newton(), m.Backtracking()]

    if method == "sfn":
        return [m.Newton(eigval_fn=lambda x: x.abs().clip(min=1e-10)), m.Backtracking()]

    if method == "inm":
        return [m.ImprovedNewton(), m.Backtracking()]

    if method == 'crn':
        return [m.CubicRegularization(m.Newton())]

    if method == "commondirections":
        return [m.SubspaceNewton(sketch_type='common_directions'), m.Backtracking()]

    if method == "trust":
        return [m.LevenbergMarquardt(m.Newton())]

    if method == "trustexact":
        return [m.TrustCG(m.Newton())]

    if method == "dogleg":
        return [m.Dogleg(m.Newton())]

    if method == "trustbfgs":
        return [m.LevenbergMarquardt(m.BFGS())]

    if method == "trustsr1":
        return [m.LevenbergMarquardt(m.SR1())]

    if method == "newtoncg":
        return [m.NewtonCG(), m.Backtracking()]

    if method == "tn":
        return [m.NewtonCG(maxiter=10), m.Backtracking()]

    if method == "trustncg":
        return [m.NewtonCGSteihaug()]

    if method == "gd":
        return [m.Backtracking()]

    if method == "cg":
        return [m.FletcherReeves(), m.StrongWolfe(c2=0.1, fallback=True)]

    if method == "bb":
        return [m.RestartOnStuck(m.BarzilaiBorwein())]

    if method == "bbstab":
        return [m.BBStab()]

    if method == "adgd":
        return [m.AdGD()]

    if method in ("gn", "gaussnewton"):
        return [m.GaussNewton(), m.Backtracking()]

    if method == "rprop":
        return [m.Rprop(alpha=1e-3)]

    if method == "lm":
        return [m.LevenbergMarquardt(m.GaussNewton())]

    if method == "mlm":
        return [m.LevenbergMarquardt(m.GaussNewton(), y=1)]

    if method == "cd":
        return [m.CD(), m.ScipyMinimizeScalar(maxiter=8)]


    raise NotImplementedError(method)