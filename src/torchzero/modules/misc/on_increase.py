import torch

from ...core import OptimizerModule


class NegateOnLossIncrease(OptimizerModule):
    """Performs an additional evaluation to check if update increases the loss. If it does,
    negates or backtracks the update.

    Args:
        backtrack (bool, optional):
            if True, sets update to minus update, otherwise sets it to zero. Defaults to True.
    """
    def __init__(self, backtrack = True):
        super().__init__({})
        self.backtrack = backtrack

    @torch.no_grad()
    def step(self, vars):
        if vars.closure is None: raise ValueError('NegateOnLossIncrease requires closure.')
        if vars.fx0 is None: vars.fx0 = vars.closure(False)

        # subtract ascent direction to params and see if loss decreases
        params = self.get_params()
        ascent_direction = vars.maybe_use_grad_(params)
        params -= ascent_direction
        vars.fx0_approx = vars.closure(False)

        # if this has no children, update params and return loss
        if self.next_module is None:
            if params is None: params = self.get_params()

            if vars.fx0_approx > vars.fx0:
                # loss increased, so we negate thea scent direction
                # we are currently at params - ascent direction
                # so we add twice the ascent direction
                params.add_(ascent_direction, alpha = 2 if self.backtrack else 1)

            # else: we are already at a lower loss point
            return vars.get_loss()

        # otherwise undo the ascent direction because it is passed to the child
        params += ascent_direction

        # if loss increases, negate ascent direction
        if vars.fx0_approx > vars.fx0:
            if self.backtrack: ascent_direction.neg_()
            else: ascent_direction.zero_()

        # otherwise undo the ascent direction and pass the updated ascent direction to the child
        return self.next_module.step(vars)


