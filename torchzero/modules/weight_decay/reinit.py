from functools import partial

import torch

from ...core import Module
from ...utils import NumberList, TensorList


def _reset_except_self(objective, modules, self: Module):
    for m in modules:
        if m is not self:
            m.reset()

class RandomReinitialize(Module):
    """On each step with probability ``p_reinit`` trigger reinitialization,
    whereby ``p_weights`` weights are reset to their initial values.

    This modifies the parameters directly. Place it as the first module.

    Args:
        p_reinit (float, optional): probability to trigger reinitialization on each step. Defaults to 0.01.
        p_weights (float, optional): probability for each weight to be set to initial value when reinitialization is triggered. Defaults to 0.1.
        store_every (int | None, optional): if set, stores new initial values every this many steps. Defaults to None.
        beta (float, optional):
            whenever ``store_every`` is triggered, uses linear interpolation with this beta.
            If ``store_every=1``, this can be set to some value close to 1 such as 0.999
            to reinitialize to slow parameter EMA. Defaults to 0.
        reset (bool, optional): whether to reset states of other modules on reinitialization. Defaults to False.
        seed (int | None, optional): random seed.
    """

    def __init__(
        self,
        p_reinit: float = 0.01,
        p_weights: float = 0.1,
        store_every: int | None = None,
        beta: float = 0,
        reset: bool = False,
        seed: int | None = None,
    ):
        defaults = dict(p_weights=p_weights, p_reinit=p_reinit, store_every=store_every, beta=beta, reset=reset, seed=seed)
        super().__init__(defaults)

    def update(self, objective):
        # this stores initial values to per-parameter states
        p_init = self.get_state(objective.params, "p_init", init="params", cls=TensorList)

        # store new params every store_every steps
        step = self.global_state.get("step", 0)
        self.global_state["step"] = step + 1

        store_every = self.defaults["store_every"]
        if (store_every is not None and step % store_every == 0):
            beta = self.get_settings(objective.params, "beta", cls=NumberList)
            p_init.lerp_(objective.params, weight=(1 - beta))

    @torch.no_grad
    def apply(self, objective):
        p_reinit = self.defaults["p_reinit"]
        device = objective.params[0].device
        generator = self.get_generator(device, self.defaults["seed"])

        # determine whether to trigger reinitialization
        reinitialize = torch.rand(1, generator=generator, device=device) < p_reinit

        # reinitialize
        if reinitialize:
            params = TensorList(objective.params)
            p_init = self.get_state(params, "p_init", init=params)


            # mask with p_weights entries being True
            p_weights = self.get_settings(params, "p_weights")
            mask = params.bernoulli_like(p_weights, generator=generator).as_bool()

            # set weights at mask to their initialization
            params.masked_set_(mask, p_init)

            # reset
            if self.defaults["reset"]:
                objective.post_step_hooks.append(partial(_reset_except_self, self=self))

        return objective