import typing as T

import torch

from ...python_tools import ScalarType
from ...tensorlist import Distributions, TensorList
from ...core import ClosureType, OptimizerModule, OptimizationState
from ._fd_formulas import _FD_Formulas


def _two_point_rcd_(closure: ClosureType, params: TensorList, perturbation: TensorList, eps: TensorList, fx0: ScalarType | None, ):
    """Two point randomized finite difference (same signature for all other finite differences functions).

    Args:
        closure (Callable): A closure that reevaluates the model and returns the loss.
        params (TensorList): TensorList with parameters.
        perturbation (TensorList): TensorList with perturbation ALREADY MULTIPLIED BY EPSILON.
        eps (TensorList): Finite difference epsilon.
        fx0 (ScalarType): Loss at fx0, to avoid reevaluating it each time. On some functions can be None when it isn't needed.

    Returns:
        TensorList with gradient estimation and approximate loss.
    """
    # positive loss
    params += perturbation
    loss_pos = closure(False)

    # negative loss
    params.sub_(perturbation, alpha = 2)
    loss_neg = closure(False)

    # restore params
    params += perturbation

    # calculate gradient estimation using central finite differences formula
    # (we square eps in denominator because perturbation is already multiplied by eps)
    # grad_est = (perturbation * (loss_pos - loss_neg)) / (2 * eps**2)
    # is equivalent to the following:
    return perturbation * eps.map(lambda x: (loss_pos - loss_neg) / (2 * x**2)), loss_pos
    # also we can't reuse the perturbatuion tensor and multiply it in place,
    # since if randomize_every is more than 1, that would break it.

def _two_point_rfd_(closure: ClosureType, params: TensorList, perturbation: TensorList, eps: TensorList, fx0: ScalarType | None):
    if fx0 is None: raise ValueError()

    params += perturbation
    fx1 = closure(False)

    params -= perturbation

    return perturbation * eps.map(lambda x: (fx1 - fx0) / x**2), fx0

def _two_point_rbd_(closure: ClosureType, params: TensorList, perturbation: TensorList, eps: TensorList, fx0: ScalarType | None):
    if fx0 is None: raise ValueError()

    params -= perturbation
    fx1 = closure(False)

    params += perturbation

    return perturbation * eps.map(lambda x: (fx0 - fx1) / x**2), fx0


class RandomizedFDM(OptimizerModule):
    """Gradient approximation via randomized finite difference.

    Args:
        eps (float, optional): finite difference epsilon. Defaults to 1e-5.
        formula (_FD_Formulas, optional): Finite difference formula. Defaults to 'forward'.
        n_samples (int, optional): number of times gradient is approximated and then averaged. Defaults to 1.
        distribution (Distributions, optional): distribution for random perturbations. Defaults to "normal".
        make_closure (bool, optional):
            if True, this makes a new closure that sets .grad attribute on each call
            with `backward = True`. If False, this simply returns the estimated gradients as the ascent direction.
        randomize_every (int, optional): number of steps between randomizing perturbations. Defaults to 1.
        randomize_closure (int, optional):
            whether to generate a new random perturbation each time closure
            is evaluated with `backward=True` (this ignores `randomize_every`). Defaults to False.
    """
    def __init__(
        self,
        eps: float = 1e-5,
        formula: _FD_Formulas = "forward",
        n_samples: int = 1,
        distribution: Distributions = "normal",
        make_closure=False,
        randomize_every: int = 1,
        randomize_closure: bool = False,
    ):
        defaults = dict(eps = eps)
        super().__init__(defaults)

        self.make_closure = make_closure

        if formula == 'forward':
            self._finite_difference = _two_point_rfd_
            self._requires_fx0 = True

        elif formula == 'backward':
            self._finite_difference = _two_point_rbd_
            self._requires_fx0 = True

        elif formula == 'central':
            self._finite_difference = _two_point_rcd_
            self._requires_fx0 = False

        else: raise ValueError(f"Unknown formula: {formula}")

        self.n_samples = n_samples
        self.distribution: Distributions = distribution
        self.randomize_every = randomize_every
        self.randomize_closure = randomize_closure

        self.perturbations = T.cast(list[TensorList], None)
        self.current_step = 0


    @torch.no_grad
    def _make_closure_step(self, state: OptimizationState, params: TensorList,epsilons: TensorList):
        if state.closure is None: raise ValueError('FDA requires a closure.')
        closure = state.closure

        # the new closure sets .grad attribute to finite difference-approximated gradients
        @torch.no_grad
        def rfdm_closure(backward = True):
            if self.randomize_closure:
                self.perturbations = [params.sample_like(epsilons, self.distribution) for _ in range(self.n_samples)]

            # closure must always evaluate the loss
            # regardless of whether we need it at fx0 or not
            loss = closure(False)

            if backward:

                if self.n_samples == 1:
                    grads, _ = self._finite_difference(closure, params, self.perturbations[0], epsilons, loss)

                else:
                    grads = params.zeros_like()
                    for i in range(self.n_samples):
                        grads += self._finite_difference(closure, params, self.perturbations[i], epsilons, loss)[0]
                    grads /= self.n_samples

                # set the grad attribute (accumulation doesn't make sense here as closure always calls zero_grad)
                for p, g in zip(params, grads):
                    p.grad = g.view_as(p)

            return loss

        # RandomizedFDM always passes the approximated gradients to its child.
        if self.next_module is None: raise ValueError("RandomizedFDM with `make_closure=True` requires a child.")
        state.closure = rfdm_closure
        return self.next_module.step(state)


    @torch.no_grad
    def _make_ascent_direction_step(self, state: OptimizationState, params: TensorList,epsilons: TensorList):
        if state.closure is None: raise ValueError('FDA requires a closure.')
        closure = state.closure

        # evaluate fx0 if it is needed for forward and backward differences.
        if state.fx0 is None and self._requires_fx0: state.fx0 = closure(False)

        if self.n_samples == 1:
            grads, state.fx0_approx = self._finite_difference(closure, params, self.perturbations[0], epsilons, state.fx0)

        else:
            grads = params.zeros_like()
            for i in range(self.n_samples):
                g, state.fx0_approx = self._finite_difference(closure, params, self.perturbations[i], epsilons, state.fx0)
                grads += g
            grads /= self.n_samples

        # FDM always passes the approximated gradients to its child.
        if self.next_module is None: raise ValueError("FDM requires a child.")
        state.ascent = grads
        return self.next_module.step(state)

    def step(self, state):
        if state.ascent is not None: raise ValueError('FDM does not accept ascent direction.')

        params = self.get_params()
        epsilons = self.get_group_key('eps')

        if self.current_step % self.randomize_every == 0 and not self.randomize_closure:
            self.perturbations = [params.sample_like(epsilons, self.distribution) for _ in range(self.n_samples)]

        self.current_step += 1

        if self.make_closure:
            return self._make_closure_step(state, params = params, epsilons=epsilons)
        else:
            return self._make_ascent_direction_step(state, params = params, epsilons=epsilons,)
