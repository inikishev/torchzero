from collections.abc import Iterable

import torch

from ...core import Chainable, Module, Var
from ...utils import TensorList

def _sequential_step(self: Module, var: Var, sequential: bool):
    params = var.params
    steps = self.settings[params[0]]['steps']

    if sequential: modules = self.get_children_sequence() * steps
    else: modules = [self.children['module']] * steps

    if var.closure is None and len(modules) > 1: raise ValueError('Multistep and Sequential require closure')

    # store original params unless this is last module and can update params directly
    params_before_steps = None if (var.is_last and var.last_module_lrs is None) else [p.clone() for p in params]

    # first step - pass var as usual
    var = modules[0].step(var)
    new_var = var

    # subsequent steps - update parameters and create new var
    if len(modules) > 1:
        for m in modules[1:]:

            # update params
            if (not new_var.skip_update):
                if new_var.last_module_lrs is not None:
                    torch._foreach_mul_(new_var.get_update(), new_var.last_module_lrs)

                torch._foreach_sub_(params, new_var.get_update())

            # create new var since we are at a new point, that means grad, update and loss will be None
            new_var = Var(params=new_var.params, closure=new_var.closure,
                            model=new_var.model, current_step=new_var.current_step + 1)

            # step
            new_var = m.step(new_var)

        # final parameter update
        if (not new_var.skip_update):
            if new_var.last_module_lrs is not None:
                torch._foreach_mul_(new_var.get_update(), new_var.last_module_lrs)

            torch._foreach_sub_(params, new_var.get_update())

    # if last module, update is applied so return new var
    if params_before_steps is None:
        new_var.stop = True
        new_var.skip_update = True
        return new_var

    # otherwise use parameter difference as update
    var.update = list(torch._foreach_sub(params_before_steps, params))
    for p, bef in zip(params, params_before_steps):
        p.set_(bef) # pyright:ignore[reportArgumentType]
    return var

class Multistep(Module):
    """Performs :code:`steps` inner steps with :code:`module` per each step.

    The update is taken to be the parameter difference between parameters before and after the inner loop."""
    def __init__(self, module: Chainable, steps: int):
        defaults = dict(steps=steps)
        super().__init__(defaults)
        self.set_child('module', module)

    @torch.no_grad
    def step(self, var):
        return _sequential_step(self, var, sequential=False)

class Sequential(Module):
    """On each step, this sequentially steps with :code:`modules` :code:`steps` times.

    The update is taken to be the parameter difference between parameters before and after the inner loop."""
    def __init__(self, modules: Iterable[Chainable], steps: int=1):
        defaults = dict(steps=steps)
        super().__init__(defaults)
        self.set_children_sequence(modules)

    @torch.no_grad
    def step(self, var):
        return _sequential_step(self, var, sequential=True)


class NegateOnLossIncrease(Module):
    """Uses an extra forward pass to evaluate loss at :code:`parameters+update`,
    if loss is larger than at :code:`parameters`,
    the update is set to 0 if :code:`backtrack=False` and to :code:`-update` otherwise"""
    def __init__(self, backtrack=False):
        defaults = dict(backtrack=backtrack)
        super().__init__(defaults=defaults)

    @torch.no_grad
    def step(self, var):
        closure = var.closure
        if closure is None: raise RuntimeError('NegateOnLossIncrease requires closure')
        backtrack = self.settings[var.params[0]]['backtrack']

        update = var.get_update()
        f_0 = var.get_loss(backward=False)

        torch._foreach_sub_(var.params, update)
        f_1 = closure(False)

        if f_1 <= f_0:
            if var.is_last and var.last_module_lrs is None:
                var.stop = True
                var.skip_update = True
                return var

            torch._foreach_add_(var.params, update)
            return var

        torch._foreach_add_(var.params, update)
        if backtrack:
            torch._foreach_neg_(var.update)
        else:
            torch._foreach_zero_(var.update)
        return var


class Online(Module):
    """Allows certain modules to be used for mini-batch optimization."""
    def __init__(self, module: Chainable,):
        super().__init__()

        self.set_child('module', module)

    @torch.no_grad
    def step(self, var):
        closure = var.closure
        if closure is None: raise ValueError("Closure must be passed for Online")
        step = self.global_state.get('step', 0) + 1
        self.global_state['step'] = step
        params = TensorList(var.params)
        p_cur = params.clone()
        p_prev = self.get_state(params, 'p_prev', cls=TensorList)
        module = self.children['module']

        if step == 1:
            var = module.step(var.clone(clone_update=False))

            p_prev.copy_(params)
            return var

        # restore previous params
        var_prev = Var(params=params, closure=closure, model=var.model, current_step=var.current_step)
        params.set_(p_prev)
        module.reset_for_online()
        module.update(var_prev)

        # restore current params
        params.set_(p_cur)
        p_prev.copy_(params)
        return module.step(var.clone(clone_update=False))
