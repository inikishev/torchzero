from ...core import OptimizerModule


def _reset_stats_hook(optimizer, state):
    for module in optimizer.unrolled_modules:
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
        first_swa (int):
            number of steps before starting PSWA, authors run PSWA starting from 40th epoch out ot 150 epochs in total.
        cycle_length (int):
            number of steps betwen updating the weight average. Authors update it once per epoch.
        num_cycles (int):
            Number of weight average updates before setting model weights to the average and proceding to the next cycle.
            Authors use 20 (meaning 20 epochs since each cycle is 1 epoch).
        cycle_start (int, optional):
            number of steps at the beginning of each SWA period before updating the weight average (default: 0).
        reset_stats (bool, optional):
            if True, when setting model parameters to SWA, resets other modules stats such as momentum velocities (default: True).
    """
    def __init__(self, pswa_start: int, cycle_length: int, num_cycles: int, cycle_start: int = 0, reset_stats:bool = True):

        super().__init__({})
        self.pswa_start = pswa_start
        self.cycle_start = cycle_start
        self.cycle_length = cycle_length
        self.num_cycles = num_cycles
        self._reset_stats = reset_stats


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
            if self._reset_stats: state.add_post_step_hook(_reset_stats_hook)

        return ret

class CyclicSWA(OptimizerModule):
    """Periodic SWA with cyclic learning rate. So it samples the weights, increases lr to `peak_lr`, samples the weights again,
    decreases lr back to `init_lr`, and samples the weights last time. Then model weights are replaced with the average of the three sampled weights,
    and next cycle starts. I made this due to a horrible misreading of the original SWA paper but it seems to work well.

    Please put this module at the end, after all other modules.

    Args:
        cswa_start (int): number of steps before starting the first CSWA cycle.
        cycle_length (int): length of each cycle in steps.
        steps_between (int): number of steps between cycles.
        init_lr (float, optional): initial and final learning rate in each cycle. Defaults to 0.
        peak_lr (float, optional): peak learning rate of each cycle. Defaults to 1.
        sample_all (float, optional): if True, instead of sampling 3 weights, it samples all weights in the cycle. Defaults to False.
        reset_stats (bool, optional):
            if True, when setting model parameters to SWA, resets other modules stats such as momentum velocities (default: True).

    """
    def __init__(self, cswa_start: int, cycle_length: int, steps_between: int, init_lr: float = 0, peak_lr: float = 1, sample_all = False, reset_stats: bool=True,):
        defaults = dict(init_lr = init_lr, peak_lr = peak_lr)
        super().__init__(defaults)
        self.cswa_start = cswa_start
        self.cycle_length = cycle_length
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.steps_between = steps_between
        self.sample_all = sample_all
        self._reset_stats = reset_stats

        self.cur = 0
        self.cycle_cur = 0
        self.n_models = 0

        self.cur_lr = self.init_lr

    def step(self, state):
        params = self.get_params()

        # start first period after `cswa_start` steps
        if self.cur >= self.cswa_start:

            ascent = state.maybe_use_grad_(params)

            # determine the lr
            point = self.cycle_cur / self.cycle_length
            init_lr, peak_lr = self.get_group_keys('init_lr', 'peak_lr')
            if point < 0.5:
                p2 = point*2
                lr = init_lr * (1-p2) + peak_lr * p2
            else:
                p2 = (1 - point)*2
                lr = init_lr * (1-p2) + peak_lr * p2

            ascent *= lr
            ret = self._update_params_or_step_with_next(state, params)

            if self.sample_all or self.cycle_cur in (0, self.cycle_length, self.cycle_length // 2):
                swa = self.get_state_key('swa')
                swa.mul_(self.n_models).add_(params).div_(self.n_models + 1)
                self.n_models += 1

                if self.cycle_cur == self.cycle_length:
                    if not self.sample_all: assert self.n_models == 3, self.n_models
                    self.n_models = 0
                    self.cycle_cur = -1

                    params.set_(swa)
                    if self._reset_stats: state.add_post_step_hook(_reset_stats_hook)

            self.cycle_cur += 1

        else:
            ret = self._update_params_or_step_with_next(state, params)

        self.cur += 1

        return ret