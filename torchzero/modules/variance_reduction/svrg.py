import warnings
from functools import partial

import torch

from ...core import Module, Objective
from ...utils import tofloat


def _reset_except_self(objective: Objective, modules, self: Module):
    for m in modules:
        if m is not self:
            m.reset()


class SVRG(Module):
    """Stochastic variance reduced gradient method (SVRG).

    To use, put SVRG as the first module, it can be used with any other modules.
    To reduce variance of a gradient estimator, put the gradient estimator before SVRG.

    First it uses first ``accum_steps`` batches to compute full gradient at initial
    parameters using gradient accumulation, the model will not be updated during this.

    Then it performs ``svrg_steps`` SVRG steps, each requires two forward and backward passes.

    After ``svrg_steps``, it goes back to full gradient computation step step.

    As an alternative to gradient accumulation you can pass "full_closure" argument to the ``step`` method,
    which should compute full gradients, set them to ``.grad`` attributes of the parameters,
    and return full loss.

    Args:
        svrg_steps (int): number of steps before calculating full gradient. This can be set to length of the dataloader.
        accum_steps (int | None, optional):
            number of steps to accumulate the gradient for. Not used if "full_closure" is passed to the ``step`` method. If None, uses value of ``svrg_steps``. Defaults to None.
        reset_before_accum (bool, optional):
            whether to reset all other modules when re-calculating full gradient. Defaults to True.
        svrg_loss (bool, optional):
            whether to replace loss with SVRG loss (calculated by same formula as SVRG gradient). Defaults to True.
        alpha (float, optional):
            multiplier to ``g_full(x_0) - g_batch(x_0)`` term, can be annealed linearly from 1 to 0 as suggested in https://arxiv.org/pdf/2311.05589#page=6

    ## Examples:
    SVRG-LBFGS
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.SVRG(len(dataloader)),
        tz.m.LBFGS(),
        tz.m.Backtracking(),
    )
    ```

    For extra variance reduction one can use Online versions of algorithms, although it won't always help.
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.SVRG(len(dataloader)),
        tz.m.Online(tz.m.LBFGS()),
        tz.m.Backtracking(),
    )

    Variance reduction can also be applied to gradient estimators.
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.SPSA(),
        tz.m.SVRG(100),
        tz.m.LR(1e-2),
    )
    ```
    ## Notes

    The SVRG gradient is computed as ``g_b(x) - alpha * (g_b(x_0) - g_f(x_0))``, where:
    - ``x`` is current parameters
    - ``x_0`` is initial parameters, where full gradient was computed
    - ``g_b`` refers to mini-batch gradient at ``x`` or ``x_0``
    - ``g_f`` refers to full gradient at ``x_0``.

    The SVRG loss is computed using the same formula.
    """
    def __init__(self, svrg_steps: int, accum_steps: int | None = None, reset_before_accum:bool=True, svrg_loss:bool=True, alpha:float=1):
        defaults = dict(svrg_steps = svrg_steps, accum_steps=accum_steps, reset_before_accum=reset_before_accum, svrg_loss=svrg_loss, alpha=alpha)
        super().__init__(defaults)


    @torch.no_grad
    def update(self, objective):
        params = objective.params
        closure = objective.closure
        assert closure is not None

        if "full_grad" not in self.global_state:

            # -------------------------- calculate full gradient ------------------------- #
            if "full_closure" in objective.storage:
                full_closure = objective.storage['full_closure']
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
                grad = objective.get_grads()
                torch._foreach_add_(accumulator, grad)

                # accumulate loss
                loss_accumulator = self.global_state.get('loss_accumulator', 0)
                loss_accumulator += tofloat(objective.loss)
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
                    objective.updates = None
                    objective.stop = True
                    objective.skip_update = True
                    return


        svrg_steps = self.defaults['svrg_steps']
        current_svrg_step = self.global_state.get('current_svrg_step', 0) + 1
        self.global_state['current_svrg_step'] = current_svrg_step

        # --------------------------- SVRG gradient closure -------------------------- #
        x0 = self.global_state['x_0']
        gf_x0 = self.global_state["full_grad"]
        ff_x0 = self.global_state['full_loss']
        use_svrg_loss = self.defaults['svrg_loss']
        alpha = self.get_settings(params, 'alpha')
        alpha_0 = alpha[0]
        if all(a == 1 for a in alpha): alpha = None

        def svrg_closure(backward=True):
            # g_b(x) - α * (g_f(x_0) - g_b(x_0)) and same for loss
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

                    # g_svrg = gb_x - alpha * (gf_x0 - gb_x0)
                    correction = torch._foreach_sub(gb_x0, gf_x0)
                    if alpha is not None: torch._foreach_mul_(correction, alpha)
                    g_svrg = torch._foreach_sub(gb_x, correction)

                    f_svrg = fb_x - alpha_0 * (fb_x0 - ff_x0)
                    for p, g in zip(params, g_svrg):
                        p.grad = g

                    if use_svrg_loss: return f_svrg
                    return fb_x

            # no backward
            if use_svrg_loss:
                fb_x = closure(False)
                torch._foreach_copy_(params, x0)
                fb_x0 = closure(False)
                torch._foreach_copy_(params, x)
                f_svrg = fb_x - alpha_0 * (fb_x0 - ff_x0)
                return f_svrg

            return closure(False)

        objective.closure = svrg_closure

        # --- after svrg_steps steps reset so that new full gradient is calculated on next step --- #
        if current_svrg_step >= svrg_steps:
            del self.global_state['current_svrg_step']
            del self.global_state['full_grad']
            del self.global_state['full_loss']
            del self.global_state['x_0']
            if self.defaults['reset_before_accum']:
                objective.post_step_hooks.append(partial(_reset_except_self, self=self))

    def apply(self, objective): return objective