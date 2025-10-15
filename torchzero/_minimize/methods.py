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
    stripped = ''.join(c for c in method.lower().strip() if c.isalnum())

    if stripped == "bfgs":
        return [m.RestartOnStuck(m.BFGS()), m.Backtracking()]

    if stripped == "lbfgs":
        return [m.LBFGS(100), m.Backtracking()]

    if stripped == "newton":
        return [m.Newton(), m.Backtracking()]

    if stripped == "sfn":
        return [m.Newton(eigval_fn=lambda x: x.abs().clip(min=1e-10)), m.Backtracking()]

    if stripped == "inm":
        return [m.ImprovedNewton(), m.Backtracking()]

    if stripped == 'crn':
        return [m.CubicRegularization(m.Newton())]

    if stripped == "commondirections":
        return [m.SubspaceNewton(sketch_type='common_directions'), m.Backtracking()]

    if stripped == "trust":
        return [m.LevenbergMarquardt(m.Newton())]

    if stripped == "dogleg":
        return [m.Dogleg(m.Newton())]

    if stripped == "trustbfgs":
        return [m.RestartOnStuck(m.LevenbergMarquardt(m.BFGS()))]

    if stripped == "trustsr1":
        return [m.RestartOnStuck(m.LevenbergMarquardt(m.SR1()))]

    if stripped == "newtoncg":
        return [m.NewtonCG(), m.Backtracking()]

    if stripped == "tn":
        return [m.NewtonCG(maxiter=10), m.Backtracking()]

    if stripped == "trustncg":
        return [m.NewtonCGSteihaug()]

    if stripped == "gd":
        return [m.Backtracking()]

    if stripped == "cg":
        return [m.FletcherReeves(), m.StrongWolfe(c2=0.1, fallback=True)]

    if stripped in ("shor", "shorr"):
        return [m.ShorR(), m.StrongWolfe(c2=0.1, fallback=True)]

    if stripped == "pgm":
        return [m.ProjectedGradientMethod(), m.StrongWolfe(c2=0.1, fallback=True)]

    if stripped == "bb":
        return [m.RestartOnStuck(m.BarzilaiBorwein())]

    if stripped == "bbstab":
        return [m.BBStab()]

    if stripped == "adgd":
        return [m.AdGD()]

    if stripped in ("bd", "bolddriver"):
        return [m.BoldDriver()]

    if stripped in ("gn", "gaussnewton"):
        return [m.GaussNewton(), m.Backtracking()]

    if stripped == "rprop":
        return [m.Rprop(alpha=1e-3)]

    if stripped == "lm":
        return [m.LevenbergMarquardt(m.GaussNewton())]

    if stripped == "mlm":
        return [m.LevenbergMarquardt(m.GaussNewton(), y=1)]

    if stripped == "cd":
        return [m.CD(), m.ScipyMinimizeScalar(maxiter=8)]

    raise NotImplementedError(stripped)