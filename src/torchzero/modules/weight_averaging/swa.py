from ...core import OptimizerModule


def _reset_stats_hook(optimizer, state):
    for module in optimizer.modules:
        module: OptimizerModule
        module.reset_stats()

class PeriodicSWA(OptimizerModule):
    """Periodic Stochastic Weight Averaging.

    Please put this module at the end, after all other modules.

    The algorithm is as follows:

    1. perform `pswa_start` normal steps before starting PSWA.

    2. Perform multiple SWA iterations. On each iteration,
    run SWA algorithm for `num_cycles` cycles,
    and set weights to the weighted average before starting the next SWA iteration.

    SWA iteration is as follows:

    1. perform `cycle_start` initial steps (can be 0)

    2. for `num_cycles`, after every `cycle_length` steps passed, update the weight average with current model weights.

    3. After `num_cycles` cycles passed, set model parameters to the weight average.

    Args:
        first_swa (int): number of steps before starting PSWA, authors run PSWA starting from 40th epoch out ot 150 epochs in total.
        cycle_length (int): number of steps betwen updating the weight average. Authors update it once per epoch.
        num_cycles (int):
            Number of weight average updates before setting model weights to the average and proceding to the next cycle.
            Authors use 20 (meaning 20 epochs since each cycle is 1 epoch).
        cycle_start (int): number of steps at the beginning of each SWA period before updating the weight average (default: 0).
    """
    def __init__(self, pswa_start: int, cycle_length: int, num_cycles: int, cycle_start: int = 0):

        super().__init__({})
        self.pswa_start = pswa_start
        self.cycle_start = cycle_start
        self.cycle_length = cycle_length
        self.num_cycles = num_cycles


        self.cur = 0
        self.period_cur = 0
        self.swa_cur = 0
        self.n_models = 0

    def step(self, state):
        swa = None
        params = self.get_params()
        ret = self._update_params_or_step_with_next(state, params)

        # start first period after `pswa_start` steps
        if self.cur >= self.pswa_start:

            # start swa after `cycle_start` steps in the current period
            if self.period_cur >= self.cycle_start:

                # swa updates on every `cycle_length`th step
                if self.swa_cur % self.cycle_length == 0:
                    swa = self.get_state_key('swa') # initialized to zeros for simplicity
                    swa.mul_(self.n_models).add_(params).div_(self.n_models + 1)
                    self.n_models += 1

                self.swa_cur += 1

            self.period_cur += 1

        self.cur += 1

        # passed num_cycles in period, set model parameters to SWA
        if self.n_models == self.num_cycles:
            self.period_cur = 0
            self.swa_cur = 0
            self.n_models = 0

            assert swa is not None # it's created above self.n_models += 1

            params.set_(swa)
            # add a hook that resets momentum, which also deletes `swa` in this module
            state.add_post_step_hook(_reset_stats_hook)

        return ret

