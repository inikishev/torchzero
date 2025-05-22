from operator import itemgetter

import torch

from .line_search import LineSearch


class TrustRegion(LineSearch):
    """Basic first order trust region, re-evaluates closure with updated parameters and scales step size based on function value change"""
    def __init__(self, nplus: float=1.5, nminus: float=0.75, c: float=1e-4, init: float = 1, backtrack: bool = True, adaptive: bool = True):
        defaults = dict(nplus=nplus, nminus=nminus, c=c, init=init, backtrack=backtrack, adaptive=adaptive)
        super().__init__(defaults)

    @torch.no_grad
    def search(self, update, vars):

        nplus, nminus, c, init, backtrack, adaptive = itemgetter('nplus','nminus','c','init','backtrack', 'adaptive')(self.settings[vars.params[0]])
        step_size = self.global_state.setdefault('step_size', init)
        previous_success = self.global_state.setdefault('previous_success', False)
        nplus_mul =  self.global_state.setdefault('nplus_mul', 1)
        nminus_mul = self.global_state.setdefault('nminus_mul', 1)


        f_0 = self.evaluate_step_size(0, vars, backward=False)

        # directional derivative (0 if c = 0 because it is not needed)
        if c == 0: d = 0
        else: d = -sum(t.sum() for t in torch._foreach_mul(vars.get_grad(), update))

        # test step size
        sufficient_f = f_0 + c * step_size * min(d, 0) # pyright:ignore[reportArgumentType]

        f_1 = self.evaluate_step_size(step_size, vars, backward=False)

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