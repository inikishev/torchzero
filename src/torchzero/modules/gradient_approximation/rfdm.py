from typing import Literal, Any, cast

import torch

from ...utils.python_tools import _ScalarLoss
from ...tensorlist import Distributions, TensorList
from ...core import _ClosureType, OptimizerModule, OptimizationVars
from ._fd_formulas import _FD_Formulas
from .base_approximator import GradientApproximatorBase

def _two_point_rcd_(closure: _ClosureType, params: TensorList, perturbation: TensorList, eps: TensorList, fx0: _ScalarLoss | None, ):
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

def _two_point_rfd_(closure: _ClosureType, params: TensorList, perturbation: TensorList, eps: TensorList, fx0: _ScalarLoss | None):
    if fx0 is None: raise ValueError()

    params += perturbation
    fx1 = closure(False)

    params -= perturbation

    return perturbation * eps.map(lambda x: (fx1 - fx0) / x**2), fx0

def _two_point_rbd_(closure: _ClosureType, params: TensorList, perturbation: TensorList, eps: TensorList, fx0: _ScalarLoss | None):
    if fx0 is None: raise ValueError()

    params -= perturbation
    fx1 = closure(False)

    params += perturbation

    return perturbation * eps.map(lambda x: (fx0 - fx1) / x**2), fx0


class RandomizedFDM(GradientApproximatorBase):
    """Gradient approximation via randomized finite difference.

    Args:
        eps (float, optional): finite difference epsilon. Defaults to 1e-5.
        formula (_FD_Formulas, optional): Finite difference formula. Defaults to 'forward'.
        n_samples (int, optional): number of times gradient is approximated and then averaged. Defaults to 1.
        distribution (Distributions, optional): distribution for random perturbations. Defaults to "normal".
        target (str, optional):
            determines what this module sets.

            "ascent" - it creates a new ascent direction but doesn't treat is as gradient.

            "grad" - it creates the gradient and sets it to `.grad` attributes (default).

            "closure" - it makes a new closure that sets the estimated gradient to the `.grad` attributes.
    """
    def __init__(
        self,
        eps: float = 1e-5,
        formula: _FD_Formulas = "forward",
        n_samples: int = 1,
        distribution: Distributions = "normal",
        target: Literal['ascent', 'grad', 'closure'] = 'grad',
    ):
        defaults = dict(eps = eps)

        if formula == 'forward':
            self._finite_difference = _two_point_rfd_
            requires_fx0 = True

        elif formula == 'backward':
            self._finite_difference = _two_point_rbd_
            requires_fx0 = True

        elif formula == 'central':
            self._finite_difference = _two_point_rcd_
            requires_fx0 = False

        else: raise ValueError(f"Unknown formula: {formula}")

        self.n_samples = n_samples
        self.distribution: Distributions = distribution

        super().__init__(defaults, requires_fx0=requires_fx0, target = target)

    @torch.no_grad
    def _make_ascent(self, closure, params, fx0):
        eps = self.get_group_key('eps')
        fx0_approx = None

        if self.n_samples == 1:
            grads, fx0_approx = self._finite_difference(closure, params, params.sample_like(eps, self.distribution), eps, fx0)

        else:
            grads = params.zeros_like()
            for i in range(self.n_samples):
                g, fx0_approx = self._finite_difference(closure, params, params.sample_like(eps, self.distribution), eps, fx0)
                grads += g
            grads /= self.n_samples

        return grads, fx0, fx0_approx