import torch

from ...core import OptimizerModule, _get_loss


class NegateOnLossIncrease(OptimizerModule):
    """Subtracts the ascent direction, and if loss didn't decrease, negates the ascent direction."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad()
    def step(self, state):
        if state.closure is None: raise ValueError('NegateOnLossIncrease requires closure.')
        if state.ascent_direction is None: raise ValueError('NegateOnLossIncrease requires ascent_direction.')
        if state.fx0 is None: state.fx0 = state.closure(False)

        # subtract ascent direction to params and see if loss decreases
        params = self.get_params()
        params -= state.ascent_direction
        state.fx0_approx = state.closure(False)

        # if this has no children, update params and return loss
        if self.child is None:
            if params is None: params = self.get_params()

            if state.fx0_approx > state.fx0:
                # loss increased, so we negate thea scent direction
                # we are currently at params - ascent direction
                # so we add twice the ascent direction
                params.add_(state.ascent_direction, alpha = 2)

            # else: we are already at a lower loss point
            return state.get_loss()

        # otherwise undo the ascent direction because it is passed to the child
        params += state.ascent_direction

        # if loss increases, negate ascent direction
        if state.fx0_approx > state.fx0:
            state.ascent_direction.neg_()

        # otherwise undo the ascent direction and pass the updated ascent direction to the child
        return self.child.step(state)



class ZeroOnLossIncrease(OptimizerModule):
    """Subtracts the ascent direction, and if loss didn't decrease, zeroes the ascent direction."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad()
    def step(self, state):
        if state.closure is None: raise ValueError('NegateOnLossIncrease requires closure.')
        if state.ascent_direction is None: raise ValueError('NegateOnLossIncrease requires ascent_direction.')
        if state.fx0 is None: state.fx0 = state.closure(False)

        # subtract ascent direction to params and see if loss decreases
        params = self.get_params()
        params -= state.ascent_direction
        fx0_approx = state.closure(False)

        # if this has no children, update params and return loss
        if self.child is None:
            if params is None: params = self.get_params()

            if fx0_approx > state.fx0:
                # loss increased, so ascent direction is zeroes
                # we are currently at params - ascent direction
                # so we just undo the step
                params.add_(state.ascent_direction)

            # else: we are already at a lower loss point
            return _get_loss(state.fx0, fx0_approx)

        # otherwise undo the ascent direction because it is passed to the child
        params += state.ascent_direction

        # if loss increases, zero ascent direction
        if fx0_approx > state.fx0:
            state.ascent_direction.zero_()

        # otherwise undo the ascent direction and pass the updated ascent direction to the child
        return self.child.step(state)