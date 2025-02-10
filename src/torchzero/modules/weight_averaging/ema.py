import torch
from ...core import OptimizerModule


def _reset_stats_hook(optimizer, state):
    for module in optimizer.unrolled_modules:
        module: OptimizerModule
        module.reset_stats()

# the reason why this needs to be at the end is ??? I NEED TO REMEMBER
class SwitchEMA(OptimizerModule):
    """Switch-EMA. Every n steps switches params to an exponential moving average of past weights.

    In the paper the switch happens after each epoch.

    Please put this module at the end, after all other modules.

    This can also function as EMA, set `update_every` to None and instead call `set_ema` and `unset_ema` on this module.


    Args:
        update_every (int): number of steps (batches) between setting model parameters to EMA.
        momentum (int): EMA momentum factor.
        reset_stats (bool, optional):
            if True, when setting model parameters to EMA, resets other modules stats such as momentum velocities.
            It might be better to set this to False if `update_every` is very small. Defaults to True.

    reference
        https://arxiv.org/abs/2402.09240
    """
    def __init__(self, update_every: int | None, momentum: float = 0.99, reset_stats: bool = True):
        defaults = dict(momentum=momentum)
        super().__init__(defaults)
        self.update_every = update_every
        self.cur_step = 0
        self.update_every = update_every
        self._reset_stats = reset_stats
        self.orig_params = None

    def set_ema(self):
        """sets module parameters to EMA, stores original parameters that can be restored by calling `unset_ema`"""
        params = self.get_params()
        self.orig_params = params.clone()
        params.set_(self.get_state_key('ema', init = 'params', params=params))

    def unset_ema(self):
        """Undoes `set_ema`."""
        if self.orig_params is None: raise ValueError('call `set_ema` first, and then `unset_ema`.')
        params = self.get_params()
        params.set_(self.orig_params)

    @torch.no_grad
    def step(self, vars):
        # if self.next_module is not None:
        #     warn(f'EMA should usually be the last module, but {self.next_module.__class__.__name__} is after it.')
        self.cur_step += 1

        params = self.get_params()
        # state.maybe_use_grad_(params)
        # update params with the child. Averaging is always applied at the end.
        ret = self._update_params_or_step_with_next(vars, params)

        ema = self.get_state_key('ema', init = 'params', params=params)
        momentum = self.get_group_key('momentum')

        ema.lerp_compat_(params, 1 - momentum)

        if (self.update_every is not None) and (self.cur_step % self.update_every == 0):
            params.set_(ema.clone())
            if self._reset_stats: vars.add_post_step_hook(_reset_stats_hook)

        return ret
