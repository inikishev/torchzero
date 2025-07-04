from operator import itemgetter

import torch

from ..line_search import LineSearchBase


class AdaptiveStepSize(LineSearchBase):
    """Basic first order step size adaptation method. Re-evaluates the function after stepping, if value decreased sufficiently,
    step size is increased. If value increased, step size is decreased.

    .. note::
        This works well in some cases, but it is often prone to collapsing.
        For a more robust alternative use :code:`tz.m.AdaptiveBacktracking`.

    Args:
        nplus (float, optional): multiplier to step size on successful steps. Defaults to 1.5.
        nminus (float, optional): multiplier to step size on unsuccessful steps. Defaults to 0.75.
        c (float, optional): descent condition. Defaults to 1e-4.
        init (float, optional): initial step size. Defaults to 1.
        backtrack (bool, optional): whether to undo the step if value increased. Defaults to True.
        adaptive (bool, optional):
            If enabled, when multiple consecutive steps have been successful or unsuccessful,
            the corresponding multipliers are increased, otherwise they are reset. Defaults to True.


    Examples:
        Adagrad with trust region:

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.Adagrad(),
                tz.m.TrustRegion()
            )

    """
    def __init__(self, nplus: float=1.5, nminus: float=0.75, c: float=1e-4, init: float = 1, backtrack: bool = True, adaptive: bool = True):
        defaults = dict(nplus=nplus, nminus=nminus, c=c, init=init, backtrack=backtrack, adaptive=adaptive)
        super().__init__(defaults)

    @torch.no_grad
    def search(self, update, var):

        nplus, nminus, c, init, backtrack, adaptive = itemgetter('nplus','nminus','c','init','backtrack', 'adaptive')(self.settings[var.params[0]])
        step_size = self.global_state.setdefault('step_size', init)
        previous_success = self.global_state.setdefault('previous_success', False)
        nplus_mul =  self.global_state.setdefault('nplus_mul', 1)
        nminus_mul = self.global_state.setdefault('nminus_mul', 1)


        f_0 = self.evaluate_step_size(0, var, backward=False)

        # directional derivative (0 if c = 0 because it is not needed)
        if c == 0: d = 0
        else: d = -sum(t.sum() for t in torch._foreach_mul(var.get_grad(), update))

        # test step size
        sufficient_f = f_0 + c * step_size * min(d, 0) # pyright:ignore[reportArgumentType]

        f_1 = self.evaluate_step_size(step_size, var, backward=False)

        proposed = step_size

        # very good step
        if f_1 < sufficient_f:
            self.global_state['step_size'] *= nplus * nplus_mul

            # two very good steps in a row - increase nplus_mul
            if adaptive:
                if previous_success: self.global_state['nplus_mul'] *= nplus
                else: self.global_state['nplus_mul'] = 1

        # acceptable step step
        #elif f_1 <= f_0: pass

        # bad step
        if f_1 >= f_0:
            self.global_state['step_size'] *= nminus * nminus_mul

            # two bad steps in a row - decrease nminus_mul
            if adaptive:
                if previous_success: self.global_state['nminus_mul'] *= nminus
                else: self.global_state['nminus_mul'] = 1

            if backtrack: proposed = 0
            else: proposed *= nminus * nminus_mul

        return proposed