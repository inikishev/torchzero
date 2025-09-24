import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast, final

import torch

from ...core import Module, Objective
from ...utils import Metrics, TensorList, safe_dict_update_, tofloat


class TerminationCriteriaBase(Module):
    def __init__(self, defaults:dict | None = None, n: int = 1):
        if defaults is None: defaults = {}
        safe_dict_update_(defaults, {"_n": n})
        super().__init__(defaults)

    @abstractmethod
    def termination_criteria(self, objective: Objective) -> bool:
        ...

    @final
    def should_terminate(self, objective: Objective) -> bool:
        n_bad = self.global_state.get('_n_bad', 0)
        n = self.defaults['_n']

        if self.termination_criteria(objective):
            n_bad += 1
            if n_bad >= n:
                self.global_state['_n_bad'] = 0
                return True

        else:
            n_bad = 0

        self.global_state['_n_bad'] = n_bad
        return False


    def update(self, objective):
        objective.should_terminate = self.should_terminate(objective)
        if objective.should_terminate: self.global_state['_n_bad'] = 0

    def apply(self, objective):
        return objective


class TerminateAfterNSteps(TerminationCriteriaBase):
    def __init__(self, steps:int):
        defaults = dict(steps=steps)
        super().__init__(defaults)

    def termination_criteria(self, objective):
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        max_steps = self.defaults['steps']
        return step >= max_steps

class TerminateAfterNEvaluations(TerminationCriteriaBase):
    def __init__(self, maxevals:int):
        defaults = dict(maxevals=maxevals)
        super().__init__(defaults)

    def termination_criteria(self, objective):
        maxevals = self.defaults['maxevals']
        assert objective.modular is not None
        return objective.modular.num_evaluations >= maxevals

class TerminateAfterNSeconds(TerminationCriteriaBase):
    def __init__(self, seconds:float, sec_fn = time.time):
        defaults = dict(seconds=seconds, sec_fn=sec_fn)
        super().__init__(defaults)

    def termination_criteria(self, objective):
        max_seconds = self.defaults['seconds']
        sec_fn = self.defaults['sec_fn']

        if 'start' not in self.global_state:
            self.global_state['start'] = sec_fn()
            return False

        seconds_passed = sec_fn() - self.global_state['start']
        return seconds_passed >= max_seconds



class TerminateByGradientNorm(TerminationCriteriaBase):
    def __init__(self, tol:float = 1e-8, n: int = 3, ord: Metrics = 2):
        defaults = dict(tol=tol, ord=ord)
        super().__init__(defaults, n=n)

    def termination_criteria(self, objective):
        tol = self.defaults['tol']
        ord = self.defaults['ord']
        return TensorList(objective.get_grads()).global_metric(ord) <= tol


class TerminateByUpdateNorm(TerminationCriteriaBase):
    """update is calculated as parameter difference"""
    def __init__(self, tol:float = 1e-8, n: int = 3, ord: Metrics = 2):
        defaults = dict(tol=tol, ord=ord)
        super().__init__(defaults, n=n)

    def termination_criteria(self, objective):
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        tol = self.defaults['tol']
        ord = self.defaults['ord']

        p_prev = self.get_state(objective.params, 'p_prev', cls=TensorList)
        if step == 0:
            p_prev.copy_(objective.params)
            return False

        should_terminate = (p_prev - objective.params).global_metric(ord) <= tol
        p_prev.copy_(objective.params)
        return should_terminate


class TerminateOnNoImprovement(TerminationCriteriaBase):
    def __init__(self, tol:float = 1e-8, n: int = 10):
        defaults = dict(tol=tol)
        super().__init__(defaults, n=n)

    def termination_criteria(self, objective):
        tol = self.defaults['tol']

        f = tofloat(objective.get_loss(False))
        if 'f_min' not in self.global_state:
            self.global_state['f_min'] = f
            return False

        f_min = self.global_state['f_min']
        d = f_min - f
        should_terminate = d <= tol
        self.global_state['f_min'] = min(f, f_min)
        return should_terminate

class TerminateOnLossReached(TerminationCriteriaBase):
    def __init__(self, value: float):
        defaults = dict(value=value)
        super().__init__(defaults)

    def termination_criteria(self, objective):
        value = self.defaults['value']
        return objective.get_loss(False) <= value

class TerminateAny(TerminationCriteriaBase):
    def __init__(self, *criteria: TerminationCriteriaBase):
        super().__init__()

        self.set_children_sequence(criteria)

    def termination_criteria(self, objective: Objective) -> bool:
        for c in self.get_children_sequence():
            if cast(TerminationCriteriaBase, c).termination_criteria(objective): return True

        return False

class TerminateAll(TerminationCriteriaBase):
    def __init__(self, *criteria: TerminationCriteriaBase):
        super().__init__()

        self.set_children_sequence(criteria)

    def termination_criteria(self, objective: Objective) -> bool:
        for c in self.get_children_sequence():
            if not cast(TerminationCriteriaBase, c).termination_criteria(objective): return False

        return True

class TerminateNever(TerminationCriteriaBase):
    def __init__(self):
        super().__init__()

    def termination_criteria(self, objective): return False

def make_termination_criteria(
    ftol: float | None = None,
    gtol: float | None = None,
    stol: float | None = None,
    maxiter: int | None = None,
    maxeval: int | None = None,
    maxsec: float | None = None,
    target_loss: float | None = None,
    extra: TerminationCriteriaBase | Sequence[TerminationCriteriaBase] | None = None,
    n: int = 3,
):
    criteria: list[TerminationCriteriaBase] = []

    if ftol is not None: criteria.append(TerminateOnNoImprovement(ftol, n=n))
    if gtol is not None: criteria.append(TerminateByGradientNorm(gtol, n=n))
    if stol is not None: criteria.append(TerminateByUpdateNorm(stol, n=n))

    if maxiter is not None: criteria.append(TerminateAfterNSteps(maxiter))
    if maxeval is not None: criteria.append(TerminateAfterNEvaluations(maxeval))
    if maxsec is not None: criteria.append(TerminateAfterNSeconds(maxsec))

    if target_loss is not None: criteria.append(TerminateOnLossReached(target_loss))

    if extra is not None:
        if isinstance(extra, TerminationCriteriaBase): criteria.append(extra)
        else: criteria.extend(extra)

    if len(criteria) == 0: return TerminateNever()
    if len(criteria) == 1: return criteria[0]
    return TerminateAny(*criteria)
