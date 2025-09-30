from collections.abc import Iterable

import torch

from ...core import Chainable, Module, Objective
from ...utils import TensorList

def _sequential_step(self: Module, objective: Objective, sequential: bool):
    params = objective.params
    steps = self.settings[params[0]]['steps']

    if sequential: modules: list[Module] = self.get_children_sequence() * steps
    else: modules = [self.children['module']] * steps

    if objective.closure is None and len(modules) > 1: raise ValueError('Multistep and Sequential require closure')

    # store original params unless this is last module and can update params directly
    params_before_steps = [p.clone() for p in params]

    # first step - pass var as usual
    objective = modules[0].step(objective)
    new_objective = objective

    # subsequent steps - update parameters and create new var
    if len(modules) > 1:
        for m in modules[1:]:

            # update params
            if (not new_objective.skip_update):
                # if new_var.last_module_lrs is not None:
                #     torch._foreach_mul_(new_var.get_update(), new_var.last_module_lrs)

                torch._foreach_sub_(params, new_objective.get_updates())

            # create new var since we are at a new point, that means grad, update and loss will be None
            new_objective = Objective(params=new_objective.params, closure=new_objective.closure,
                            model=new_objective.model, current_step=new_objective.current_step + 1)

            # step
            new_objective = m.step(new_objective)

        # final parameter update
        if (not new_objective.skip_update):
            # if new_var.last_module_lrs is not None:
            #     torch._foreach_mul_(new_var.get_update(), new_var.last_module_lrs)

            torch._foreach_sub_(params, new_objective.get_updates())

    # if last module, update is applied so return new var
    # if params_before_steps is None:
    #     new_var.stop = True
    #     new_var.skip_update = True
    #     return new_var

    # otherwise use parameter difference as update
    objective.updates = list(torch._foreach_sub(params_before_steps, params))
    for p, bef in zip(params, params_before_steps):
        p.set_(bef) # pyright:ignore[reportArgumentType]
    return objective

class Multistep(Module):
    """Performs ``steps`` inner steps with ``module`` per each step.

    The update is taken to be the parameter difference between parameters before and after the inner loop."""
    def __init__(self, module: Chainable, steps: int):
        defaults = dict(steps=steps)
        super().__init__(defaults)
        self.set_child('module', module)

    @torch.no_grad
    def apply(self, objective):
        return _sequential_step(self, objective, sequential=False)

class Sequential(Module):
    """On each step, this sequentially steps with ``modules`` ``steps`` times.

    The update is taken to be the parameter difference between parameters before and after the inner loop."""
    def __init__(self, modules: Iterable[Chainable], steps: int=1):
        defaults = dict(steps=steps)
        super().__init__(defaults)
        self.set_children_sequence(modules)

    @torch.no_grad
    def apply(self, objective):
        return _sequential_step(self, objective, sequential=True)


class NegateOnLossIncrease(Module):
    """Uses an extra forward pass to evaluate loss at ``parameters+update``,
    if loss is larger than at ``parameters``,
    the update is set to 0 if ``backtrack=False`` and to ``-update`` otherwise"""
    def __init__(self, backtrack=False):
        defaults = dict(backtrack=backtrack)
        super().__init__(defaults=defaults)

    @torch.no_grad
    def apply(self, objective):
        closure = objective.closure
        if closure is None: raise RuntimeError('NegateOnLossIncrease requires closure')
        backtrack = self.defaults['backtrack']

        update = objective.get_updates()
        f_0 = objective.get_loss(backward=False)

        torch._foreach_sub_(objective.params, update)
        f_1 = closure(False)

        if f_1 <= f_0:
            # if var.is_last and var.last_module_lrs is None:
            #     var.stop = True
            #     var.skip_update = True
            #     return var

            torch._foreach_add_(objective.params, update)
            return objective

        torch._foreach_add_(objective.params, update)
        if backtrack:
            torch._foreach_neg_(objective.updates)
        else:
            torch._foreach_zero_(objective.updates)
        return objective


class Online(Module):
    """Allows certain modules to be used for mini-batch optimization.

    Examples:

    Online L-BFGS with Backtracking line search
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Online(tz.m.LBFGS()),
        tz.m.Backtracking()
    )
    ```

    Online L-BFGS trust region
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.TrustCG(tz.m.Online(tz.m.LBFGS()))
    )
    ```

    """
    def __init__(self, module: Module,):
        super().__init__()
        self.set_child('module', module)

    @torch.no_grad
    def update(self, objective):
        closure = objective.closure
        if closure is None: raise ValueError("Closure must be passed for Online")

        step = self.global_state.get('step', 0) + 1
        self.global_state['step'] = step

        params = TensorList(objective.params)
        p_cur = params.clone()
        p_prev = self.get_state(params, 'p_prev', cls=TensorList)

        module = self.children['module']
        var_c = objective.clone(clone_updates=False)

        # on 1st step just step and store previous params
        if step == 1:
            p_prev.copy_(params)

            module.update(var_c)
            objective.update_attrs_from_clone_(var_c)
            return

        # restore previous params and update
        prev_objective = Objective(params=params, closure=closure, model=objective.model, current_step=objective.current_step)
        params.set_(p_prev)
        module.reset_for_online()
        module.update(prev_objective)

        # restore current params and update
        params.set_(p_cur)
        p_prev.copy_(params)
        module.update(var_c)
        objective.update_attrs_from_clone_(var_c)

    @torch.no_grad
    def apply(self, objective):
        module = self.children['module']
        return module.apply(objective.clone(clone_updates=False))

    def get_H(self, objective):
        return self.children['module'].get_H(objective)