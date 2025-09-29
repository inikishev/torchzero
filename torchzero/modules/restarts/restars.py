from abc import ABC, abstractmethod
from functools import partial
from typing import final, Literal, cast

import torch

from ...core import Chainable, Module, Objective
from ...utils import TensorList
from ..termination import TerminationCriteriaBase

def _reset_except_self(objective, modules, self: Module):
    for m in modules:
        if m is not self:
            m.reset()

class RestartStrategyBase(Module, ABC):
    """Base class for restart strategies.

    On each ``update``/``step`` this checks reset condition and if it is satisfied,
    resets the modules before updating or stepping.
    """
    def __init__(self, defaults: dict | None = None, modules: Chainable | None = None):
        if defaults is None: defaults = {}
        super().__init__(defaults)
        if modules is not None:
            self.set_child('modules', modules)

    @abstractmethod
    def should_reset(self, objective: Objective) -> bool:
        """returns whether reset should occur"""

    def _reset_on_condition(self, objective: Objective):
        modules = self.children.get('modules', None)

        if self.should_reset(objective):
            if modules is None:
                objective.post_step_hooks.append(partial(_reset_except_self, self=self))
            else:
                modules.reset()

        return modules

    @final
    def update(self, objective):
        modules = self._reset_on_condition(objective)
        if modules is not None:
            modules.update(objective)

    @final
    def apply(self, objective):
        # don't check here because it was check in `update`
        modules = self.children.get('modules', None)
        if modules is None: return objective
        return modules.apply(objective.clone(clone_updates=False))

    @final
    def step(self, objective):
        modules = self._reset_on_condition(objective)
        if modules is None: return objective
        return modules.step(objective.clone(clone_updates=False))



class RestartOnStuck(RestartStrategyBase):
    """Resets the state when update (difference in parameters) is zero for multiple steps in a row.

    Args:
        modules (Chainable | None):
            modules to reset. If None, resets all modules.
        tol (float, optional):
            step is considered failed when maximum absolute parameter difference is smaller than this. Defaults to None (uses twice the smallest respresentable number)
        n_tol (int, optional):
            number of failed consequtive steps required to trigger a reset. Defaults to 10.

    """
    def __init__(self, modules: Chainable | None, tol: float | None = None, n_tol: int = 10):
        defaults = dict(tol=tol, n_tol=n_tol)
        super().__init__(defaults, modules)

    @torch.no_grad
    def should_reset(self, objective):
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        params = TensorList(objective.params)
        tol = self.defaults['tol']
        if tol is None: tol = torch.finfo(params[0].dtype).tiny * 2
        n_tol = self.defaults['n_tol']
        n_bad = self.global_state.get('n_bad', 0)

        # calculate difference in parameters
        prev_params = self.get_state(params, 'prev_params', cls=TensorList)
        update = params - prev_params
        prev_params.copy_(params)

        # if update is too small, it is considered bad, otherwise n_bad is reset to 0
        if step > 0:
            if update.abs().global_max() <= tol:
                n_bad += 1

            else:
                n_bad = 0

        self.global_state['n_bad'] = n_bad

        # no progress, reset
        if n_bad >= n_tol:
            self.global_state.clear()
            return True

        return False


class RestartEvery(RestartStrategyBase):
    """Resets the state every n steps

    Args:
        modules (Chainable | None):
            modules to reset. If None, resets all modules.
        steps (int | Literal["ndim"]):
            number of steps between resets. "ndim" to use number of parameters.
    """
    def __init__(self, modules: Chainable | None, steps: int | Literal['ndim']):
        defaults = dict(steps=steps)
        super().__init__(defaults, modules)

    def should_reset(self, objective):
        step = self.global_state.get('step', 0) + 1
        self.global_state['step'] = step

        n = self.defaults['steps']
        if isinstance(n, str): n = sum(p.numel() for p in objective.params if p.requires_grad)

        # reset every n steps
        if step % n == 0:
            self.global_state.clear()
            return True

        return False

class RestartOnTerminationCriteria(RestartStrategyBase):
    def __init__(self, modules: Chainable | None, criteria: "TerminationCriteriaBase"):
        super().__init__(None, modules)
        self.set_child('criteria', criteria)

    def should_reset(self, objective):
        criteria = cast(TerminationCriteriaBase, self.children['criteria'])
        return criteria.should_terminate(objective)

class PowellRestart(RestartStrategyBase):
    """Powell's two restarting criterions for conjugate gradient methods.

    The restart clears all states of ``modules``.

    Args:
        modules (Chainable | None):
            modules to reset. If None, resets all modules.
        cond1 (float | None, optional):
            criterion that checks for nonconjugacy of the search directions.
            Restart is performed whenevr g^Tg_{k+1} >= cond1*||g_{k+1}||^2.
            The default condition value of 0.2 is suggested by Powell. Can be None to disable that criterion.
        cond2 (float | None, optional):
            criterion that checks if direction is not effectively downhill.
            Restart is performed if -1.2||g||^2 < d^Tg < -0.8||g||^2.
            Defaults to 0.2. Can be None to disable that criterion.

    Reference:
        Powell, Michael James David. "Restart procedures for the conjugate gradient method." Mathematical programming 12.1 (1977): 241-254.
    """
    def __init__(self, modules: Chainable | None, cond1:float | None = 0.2, cond2:float | None = 0.2):
        defaults=dict(cond1=cond1, cond2=cond2)
        super().__init__(defaults, modules)

    def should_reset(self, objective):
        g = TensorList(objective.get_grads())
        cond1 = self.defaults['cond1']; cond2 = self.defaults['cond2']

        # -------------------------------- initialize -------------------------------- #
        if 'initialized' not in self.global_state:
            self.global_state['initialized'] = 0
            g_prev = self.get_state(objective.params, 'g_prev', init=g)
            return False

        g_g = g.dot(g)

        reset = False
        # ------------------------------- 1st condition ------------------------------ #
        if cond1 is not None:
            g_prev = self.get_state(objective.params, 'g_prev', must_exist=True, cls=TensorList)
            g_g_prev = g_prev.dot(g)

            if g_g_prev.abs() >= cond1 * g_g:
                reset = True

        # ------------------------------- 2nd condition ------------------------------ #
        if (cond2 is not None) and (not reset):
            d_g = TensorList(objective.get_updates()).dot(g)
            if (-1-cond2) * g_g < d_g < (-1 + cond2) * g_g:
                reset = True

        # ------------------------------ clear on reset ------------------------------ #
        if reset:
            self.global_state.clear()
            self.clear_state_keys('g_prev')
            return True

        return False

# this requires direction from inner module,
# so both this and inner have to be a single module
class BirginMartinezRestart(Module):
    """the restart criterion for conjugate gradient methods designed by Birgin and Martinez.

    This criterion restarts when when the angle between dk+1 and −gk+1 is not acute enough.

    The restart clears all states of ``module``.

    Args:
        module (Module):
            module to restart, should be a conjugate gradient or possibly a quasi-newton method.
        cond (float, optional):
            Restart is performed whenevr d^Tg > -cond*||d||*||g||.
            The default condition value of 1e-3 is suggested by Birgin and Martinez.

    Reference:
        Birgin, Ernesto G., and José Mario Martínez. "A spectral conjugate gradient method for unconstrained optimization." Applied Mathematics & Optimization 43.2 (2001): 117-128.
    """
    def __init__(self, module: Module, cond:float = 1e-3):
        defaults=dict(cond=cond)
        super().__init__(defaults)

        self.set_child("module", module)

    def update(self, objective):
        module = self.children['module']
        module.update(objective)

    def apply(self, objective):
        module = self.children['module']
        objective = module.apply(objective.clone(clone_updates=False))

        cond = self.defaults['cond']
        g = TensorList(objective.get_grads())
        d = TensorList(objective.get_updates())
        d_g = d.dot(g)
        d_norm = d.global_vector_norm()
        g_norm = g.global_vector_norm()

        # d in our case is same direction as g so it has a minus sign
        if -d_g > -cond * d_norm * g_norm:
            module.reset()
            objective.updates = g.clone()
            return objective

        return objective
