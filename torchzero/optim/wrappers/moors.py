from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import torch

from ...utils import TensorList
from .wrapper import WrapperBase

Closure = Callable[[bool], Any]

class MoorsWrapper(WrapperBase):
    """Use moo-rs (pymoors) is PyTorch optimizer.

    Note that this performs full minimization on each step,
    so usually you would want to perform a single step.

    To use this, define a function that accepts fitness function and number of variables and returns a pymoors algorithm:

    ```python
    alg_fn = lambda fitness_fn, num_vars: pymoors.Nsga2(
        fitness_fn=fitness_fn,
        num_vars=num_vars,
        num_iterations=100,
        sampler = pymoors.RandomSamplingFloat(min=-3, max=3),
        crossover = pymoors.SinglePointBinaryCrossover(),
        mutation = pymoors.GaussianMutation(gene_mutation_rate=1e-2, sigma=0.1),
        population_size = 32,
        num_offsprings = 32,
    )

    optimizer = MoorsWrapper(model.parameters(), alg_fn)
    ```

    All algorithms in pymoors have slightly different APIs, refer to their docs.

    """
    def __init__(
        self,
        params,
        algorithm_fn: Callable[[Callable[[np.ndarray], np.ndarray], int], Any]
    ):
        super().__init__(params, {})
        self._algorithm_fn = algorithm_fn

    def _objective(self, x: np.ndarray, params, closure):
        fs = []
        for x_i in x:
            f_i = self._fs(x_i, params=params, closure=closure)
            fs.append(f_i)
        return np.stack(fs, dtype=np.float64) # pymoors needs float64

    @torch.no_grad
    def step(self, closure: Closure):
        params = TensorList(self._get_params())
        objective = partial(self._objective, params=params, closure=closure)

        algorithm = self._algorithm_fn(objective, params.global_numel())

        algorithm.run()
        pop = algorithm.population

        params.from_vec_(torch.as_tensor(pop.best[0].genes, device = params[0].device, dtype=params[0].dtype,))
        return pop.best[0].fitness

