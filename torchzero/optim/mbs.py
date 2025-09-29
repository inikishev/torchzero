from typing import NamedTuple
import math
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal

import numpy as np


def format_number(number, n):
    """Rounds to n significant digits after the decimal point."""
    if number == 0: return 0
    if math.isnan(number) or math.isinf(number) or (not math.isfinite(number)): return number
    if n <= 0: raise ValueError("n must be positive")

    dec = Decimal(str(number))
    if dec.is_zero(): return 0
    if number > 10**n or dec % 1 == 0: return int(dec)

    if abs(dec) >= 1:
        places = n
    else:
        frac_str = format(abs(dec), 'f').split('.')[1]
        leading_zeros = len(frac_str) - len(frac_str.lstrip('0'))
        places = leading_zeros + n

    quantizer = Decimal('1e-' + str(places))
    rounded_dec = dec.quantize(quantizer, rounding=ROUND_HALF_UP)

    if rounded_dec % 1 == 0: return int(rounded_dec)
    return float(rounded_dec)

def _nonfinite_to_inf(x):
    if not math.isfinite(x): return math.inf
    return x

def _tofloatlist(x) -> list[float]:
    if isinstance(x, (int,float)): return [x]
    if isinstance(x, np.ndarray) and x.size == 1: return [float(x.item())]
    return [float(i) for i in x]

class Trial(NamedTuple):
    x: float
    f: tuple[float, ...]

class Solution(NamedTuple):
    x: float
    f: tuple[float, ...]
    trials: list[Trial]

class MBS:
    """Univariate minimization via grid search followed by refining, supports multi-objective functions.

    This tends to outperform bayesian optimization for learning rate tuning, it is also good for plotting.

    First it evaluates all points defined in ``grid``. The grid doesn't have to be dense and the solution doesn't
    have to be between the endpoints.

    Then it picks ``num_candidates`` best points per each objective. If any of those points are endpoints,
    it expands the search space by ``step`` in that direction and evaluates the new endpoint.

    Otherwise it keeps picking points between best points and evaluating them, until ``num_binary`` evaluations
    have been performed.

    Args:
        grid (Iterable[float], optional): values for initial grid search. If ``log_scale=True``, should be in log10 scale.
        step (float, optional): expansion step size. Defaults to 1.
        num_candidates (int, optional): number of best points to sample new points around on each iteration. Defaults to 2.
        num_binary (int, optional): maximum number of new points sampled via binary search. Defaults to 7.
        num_expansions (int, optional): maximum number of expansions (not counted towards binary search points). Defaults to 7.
        rounding (int, optional): rounding is to significant digits, avoids evaluating points that are too close.
        lb (float | None, optional): lower bound. If ``log_scale=True``, should be in log10 scale.
        ub (float | None, optional): upper bound. If ``log_scale=True``, should be in log10 scale.
        log_scale (bool, optional):
            whether to minimize in log10 scale. If true, it is assumed that
            ``grid``, ``lb`` and ``ub`` are given in log10 scale.

    Example:

    ```python
    def objective(x: float):
        x = x * 4
        return -(np.sin(x) * (x / 3) + np.cos(x*2.5) * 2 - 0.05 * (x-5)**2)

    mbs = MBS(grid=[-1, 0, 1, 2, 3, 4], step=1, num_binary=10, num_expansions=10)

    x, f, trials = mbs.run(objective)
    # x - solution
    # f - value at solution x
    # trials - list of trials, each trial is a named tuple: Trial(x, f)
    """

    def __init__(
        self,
        grid: Iterable[float],
        step: float,
        num_candidates: int = 3,
        num_binary: int = 20,
        num_expansions: int = 20,
        rounding: int| None = 2,
        lb = None,
        ub = None,
        log_scale: bool = False,
    ):
        self.objectives: dict[int, dict[float,float]] = {}
        """dictionary of objectives, each maps point (x) to value (v)"""

        self.evaluated: set[float] = set()
        """set of evaluated points (x)"""

        grid = tuple(grid)
        if len(grid) == 0: raise ValueError("At least one grid search point must be specified")
        self.grid = sorted(grid)

        self.step = step
        self.num_candidates = num_candidates
        self.num_binary = num_binary
        self.num_expansions = num_expansions
        self.rounding = rounding
        self.log_scale = log_scale
        self.lb = lb
        self.ub = ub

    def _get_best_x(self, n: int, objective: int):
        """n best points"""
        obj = self.objectives[objective]
        v_to_x = [(v,x) for x,v in obj.items()]
        v_to_x.sort(key = lambda vx: vx[0])
        xs = [x for v,x in v_to_x]
        return xs[:n]

    def _suggest_points_around(self, x: float, objective: int):
        """suggests points around x"""
        points = list(self.objectives[objective].keys())
        points.sort()
        if x not in points: raise RuntimeError(f"{x} not in {points}")

        expansions = []
        if x == points[0]:
            expansions.append((x-self.step, 'expansion'))

        if x == points[-1]:
            expansions.append((x+self.step, 'expansion'))

        if len(expansions) != 0: return expansions

        idx = points.index(x)
        xm = points[idx-1]
        xp = points[idx+1]

        x1 = (x - (x - xm)/2)
        x2 = (x + (xp - x)/2)

        return [(x1, 'binary'), (x2, 'binary')]

    def _out_of_bounds(self, x):
        if self.lb is not None and x < self.lb: return True
        if self.ub is not None and x > self.ub: return True
        return False

    def _evaluate(self, fn, x):
        """Evaluate a point, returns False if point is already in history"""
        if self.rounding is not None: x = format_number(x, self.rounding)
        if x in self.evaluated: return False
        if self._out_of_bounds(x): return False

        self.evaluated.add(x)

        if self.log_scale: vals = _tofloatlist(fn(10 ** x))
        else: vals = _tofloatlist(fn(x))
        vals = [_nonfinite_to_inf(v) for v in vals]

        for idx, v in enumerate(vals):
            if idx not in self.objectives: self.objectives[idx] = {}
            self.objectives[idx][x] = v

        return True

    def run(self, fn) -> Solution:
        # step 1 - gr id search
        for x in self.grid:
            self._evaluate(fn, x)

        # step 2 - binary search
        while True:
            if (self.num_candidates <= 0) or (self.num_expansions <= 0 and self.num_binary <= 0): break

            # suggest candidates
            candidates: list[tuple[float, str]] = []

            # sample around best points
            for objective in self.objectives:
                best_points = self._get_best_x(self.num_candidates, objective)
                for p in best_points:
                    candidates.extend(self._suggest_points_around(p, objective=objective))

            # filter
            if self.num_expansions <= 0:
                candidates = [(x,t) for x,t in candidates if t != 'expansion']

            if self.num_candidates <= 0:
                candidates = [(x,t) for x,t in candidates if t != 'binary']

            # if expansion was suggested, discard anything else
            types = [t for x, t in candidates]
            if any(t == 'expansion' for t in types):
                candidates = [(x,t) for x,t in candidates if t == 'expansion']

            # evaluate candidates
            terminate = False
            at_least_one_evaluated = False
            for x, t in candidates:
                evaluated = self._evaluate(fn, x)
                if not evaluated: continue
                at_least_one_evaluated = True

                if t == 'expansion': self.num_expansions -= 1
                elif t == 'binary': self.num_binary -= 1

                if self.num_binary < 0:
                    terminate = True
                    break

            if terminate: break
            if not at_least_one_evaluated:
                if self.rounding is None: break
                self.rounding += 1
                if self.rounding == 100: break

        # create dict[float, tuple[float,...]]
        ret = {}
        for i, objective in enumerate(self.objectives.values()):
            for x, v in objective.items():
                if self.log_scale: x = 10 ** x
                if x not in ret: ret[x] = [None for _ in self.objectives]
                ret[x][i] = v

        for v in ret.values():
            assert len(v) == len(self.objectives), v
            assert all(i is not None for i in v), v

        # ret maps x to list of per-objective values, e.g. {1: [0.1, 0.3], ...}
        # now make a list of trials as they are easier to work with
        trials: list[Trial] = []
        for x, values in ret.items():
            trials.append(Trial(x=x, f=values))

        # sort trials by sum of values
        trials.sort(key = lambda trial: sum(trial.f))
        return Solution(x=trials[0].x, f=trials[0].f, trials=trials)

def mbs_minimize(
    fn,
    grid: Iterable[float],
    step: float,
    num_candidates: int = 3,
    num_binary: int = 20,
    num_expansions: int = 20,
    rounding=2,
    lb:float | None = None,
    ub:float | None = None,
    log_scale=False,
) -> Solution:
    """minimize univariate function via MBS.

    Args:
        fn (function): objective function that accepts a float and returns a float or a sequence of floats to minimize.
        step (float, optional): expansion step size. Defaults to 1.
        num_candidates (int, optional): number of best points to sample new points around on each iteration. Defaults to 2.
        num_binary (int, optional): maximum number of new points sampled via binary search. Defaults to 7.
        num_expansions (int, optional): maximum number of expansions (not counted towards binary search points). Defaults to 7.
        rounding (int, optional): rounding is to significant digits, avoids evaluating points that are too close.
        lb (float | None, optional): lower bound. If ``log_scale=True``, should be in log10 scale.
        ub (float | None, optional): upper bound. If ``log_scale=True``, should be in log10 scale.
        log_scale (bool, optional):
            whether to minimize in log10 scale. If true, it is assumed that
            ``grid``, ``lb`` and ``ub`` are given in log10 scale.

    Example:

    ```python
    def objective(x: float):
        x = x * 4
        return -(np.sin(x) * (x / 3) + np.cos(x*2.5) * 2 - 0.05 * (x-5)**2)

    x, f, trials = mbs_minimize(objective, grid=[-1, 0, 1, 2, 3, 4], step=1, num_binary=10, num_expansions=10)
    # x - solution
    # f - value at solution x
    # trials - list of trials, each trial is a named tuple: Trial(x, f)
    """
    mbs = MBS(grid, step=step, num_candidates=num_candidates, num_binary=num_binary, num_expansions=num_expansions, rounding=rounding, lb=lb, ub=ub, log_scale=log_scale)
    return mbs.run(fn)
