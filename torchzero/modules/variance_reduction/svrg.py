import warnings
from functools import partial

import torch

from ...core.module import Module
from ...utils import tofloat


def _reset_except_self(optimizer, var, self: Module):
    for m in optimizer.unrolled_modules:
        if m is not self:
            m.reset()

class SVRG(Module):
    """either pass "full_closure" to step or set n_steps"""
    def __init__(self, svrg_steps: int, accum_steps: int | None = None, reset_before_accum:bool=True):
        defaults = dict(svrg_steps = svrg_steps, accum_steps=accum_steps, reset_before_accum=reset_before_accum)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        params = var.params
        closure = var.closure
        assert closure is not None

        if "full_grad" not in self.global_state:

            # -------------------------- calculate full gradient ------------------------- #
            if "full_closure" in var.storage:
                full_closure = var.storage['full_closure']
                with torch.enable_grad():
                    full_loss = full_closure()
                    if all(p.grad is None for p in params):
                        warnings.warn("all gradients are None after evaluating full_closure.")

                    full_grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                    self.global_state["full_loss"] = full_loss
                    self.global_state["full_grad"] = full_grad
                    self.global_state['x_0'] = [p.clone() for p in params]

                # current batch will be used for svrg update

            else:
                # accumulate gradients over n steps
                accum_steps = self.defaults['accum_steps']
                if accum_steps is None: accum_steps = self.defaults['svrg_steps']

                current_accum_step = self.global_state.get('current_accum_step', 0) + 1
                self.global_state['current_accum_step'] = current_accum_step

                # accumulate grads
                accumulator = self.get_state(params, 'accumulator')
                grad = var.get_grad()
                torch._foreach_add_(accumulator, grad)

                # accumulate loss
                loss_accumulator = self.global_state.get('loss_accumulator', 0)
                loss_accumulator += tofloat(var.loss)
                self.global_state['loss_accumulator'] = loss_accumulator

                # on nth step, use the accumulated gradient
                if current_accum_step >= accum_steps:
                    torch._foreach_div_(accumulator, accum_steps)
                    self.global_state["full_grad"] = accumulator
                    self.global_state["full_loss"] = loss_accumulator / accum_steps

                    self.global_state['x_0'] = [p.clone() for p in params]
                    self.clear_state_keys('accumulator')
                    del self.global_state['current_accum_step']

                # otherwise skip update until enough grads are accumulated
                else:
                    var.update = None
                    var.stop = True
                    var.skip_update = True
                    return var


        svrg_steps = self.defaults['svrg_steps']
        current_svrg_step = self.global_state.get('current_svrg_step', 0) + 1
        self.global_state['current_svrg_step'] = current_svrg_step

        # --------------------------- SVRG gradient closure -------------------------- #
        x0 = self.global_state['x_0']
        gf_x0 = self.global_state["full_grad"]
        ff_x0 = self.global_state['full_loss']

        def svrg_closure(backward=True):
            # g_b(x) + g_f(x_0) - g_b(x_0) and same for loss
            with torch.no_grad():
                x = [p.clone() for p in params]

                if backward:
                    # f and g at x
                    with torch.enable_grad(): fb_x = closure()
                    gb_x = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

                    # f and g at x_0
                    torch._foreach_copy_(params, x0)
                    with torch.enable_grad(): fb_x0 = closure()
                    gb_x0 = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                    torch._foreach_copy_(params, x)

                    # g_svrg = gb_x + gf_x0 - gb_x0
                    g_svrg = torch._foreach_add(gb_x, gf_x0)
                    torch._foreach_sub_(g_svrg, gb_x0)

                    f_svrg = fb_x + ff_x0 - fb_x0
                    for p, g in zip(params, g_svrg):
                        p.grad = g

                    return f_svrg

            # no backward
            fb_x = closure(False)
            torch._foreach_copy_(params, x0)
            fb_x0 = closure(False)
            torch._foreach_copy_(params, x)
            f_svrg = fb_x + ff_x0 - fb_x0
            return f_svrg

        var.closure = svrg_closure

        # --- after svrg_steps steps reset so that new full gradient is calculated on next step --- #
        if current_svrg_step >= svrg_steps:
            del self.global_state['current_svrg_step']
            del self.global_state['full_grad']
            del self.global_state['full_loss']
            del self.global_state['x_0']
            if self.defaults['reset_before_accum']:
                var.post_step_hooks.append(partial(_reset_except_self, self=self))

        return var
