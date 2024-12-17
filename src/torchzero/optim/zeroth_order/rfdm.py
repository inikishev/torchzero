import typing as T
from collections import abc

import torch

from ...tensorlist import Distributions
from ...modules import SGD, OptimizerWrapper
from ...modules import RandomizedFDM as _RandomizedFDM
from ...modules.gradient_approximation._fd_formulas import _FD_Formulas
from ..modular import ModularOptimizer


class RandomizedFDM(ModularOptimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        eps: float = 1e-3,
        momentum:float = 0,
        weight_decay:float = 0,
        dampening: float = 0,
        nesterov:bool = False,
        formula: _FD_Formulas = "forward",
        n_samples: int = 1,
        distribution: Distributions = "normal",
        randomize_every: int = 1,
    ):
        """Randomized finite difference gradient approximation (e.g. SPSA, RDSA, Nesterov random search).

        With `forward` and `backward` formulas performs `1 + n_samples` evaluations per step;
        with `central` formula performs `2 * n_samples` evaluations per step.

        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): learning rate. Defaults to 1e-3.
            eps (float, optional): finite difference epsilon. Defaults to 1e-3.
            momentum (float, optional): momentum factor. Defaults to 0.
            weight_decay (float, optional): weight decay (L2 penalty). Defaults to 0.
            dampening (float, optional): dampening for momentum. Defaults to 0.
            nesterov (bool, optional): enables Nesterov momentum (supports dampening). Defaults to False.
            formula (_FD_Formulas, optional): finite difference formula. Defaults to "forward".
            n_samples (int, optional): number of random gradient approximations that will be averaged. Defaults to 1.
            distribution (Distributions, optional): distribution for random perturbations. Defaults to "normal".
            randomize_every (int, optional): number of steps between randomizing perturbations. Defaults to 1.
        """
        modules = [
            _RandomizedFDM(
                eps=eps,
                formula=formula,
                n_samples=n_samples,
                distribution=distribution,
                randomize_every=randomize_every,
            ),
            SGD(lr = lr, momentum = momentum, weight_decay = weight_decay, dampening = dampening, nesterov = nesterov)
        ]
        super().__init__(params, modules)


class SPSA(RandomizedFDM):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        eps: float = 1e-3,
        momentum:float = 0,
        weight_decay:float = 0,
        dampening: float = 0,
        nesterov:bool = False,
        formula: _FD_Formulas = "central",
        n_samples: int = 1,
        distribution: Distributions = 'rademacher',
        randomize_every: int = 1,
    ):
        """Simultaneous perturbation stochastic approximation method.
        This is the same as a randomized finite difference method with central formula
        and perturbations taken from rademacher distibution.
        Due to rademacher having values -1 or 1, the original formula divides by the perturbation,
        but that is equivalent to multiplying by it, which is the same as central difference formula.

        Spall, J. C. (1992), “Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation,” IEEE Transactions on Automatic Control, vol. 37(3), pp. 332–341.

        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): learning rate. Defaults to 1e-3.
            eps (float, optional): finite difference epsilon. Defaults to 1e-3.
            momentum (float, optional): momentum factor. Defaults to 0.
            weight_decay (float, optional): weight decay (L2 penalty). Defaults to 0.
            dampening (float, optional): dampening for momentum. Defaults to 0.
            nesterov (bool, optional): enables Nesterov momentum (supports dampening). Defaults to False.
            formula (_FD_Formulas, optional): finite difference formula. Defaults to "central".
            n_samples (int, optional): number of random gradient approximations that will be averaged. Defaults to 1.
            distribution (Distributions, optional): distribution for random perturbations. Defaults to "rademacher".
            randomize_every (int, optional): number of steps between randomizing perturbations. Defaults to 1.
        """
        super().__init__(
            params = params,
            lr = lr,
            eps = eps,
            momentum = momentum,
            weight_decay = weight_decay,
            dampening = dampening,
            nesterov = nesterov,
            formula = formula,
            n_samples = n_samples,
            distribution = distribution,
            randomize_every=randomize_every,
        )


class RandomGaussianSmoothing(RandomizedFDM):
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        eps: float = 1e-2,
        momentum:float = 0,
        weight_decay:float = 0,
        dampening: float = 0,
        nesterov:bool = False,
        formula: _FD_Formulas = "forward",
        n_samples: int = 10,
        distribution: Distributions = 'normal',
        randomize_every: int = 1,
    ):
        """Random search with gaussian smoothing.
        This is similar to forward randomized finite difference method, and it
        approximates and averages the gradient with multiple random perturbations taken from normal distribution,
        which is an approximation for the gradient of a gaussian smoothed version of the objective function.

        Nesterov, Y., & Spokoiny, V. (2017). Random gradient-free minimization of convex functions. Foundations of Computational Mathematics, 17(2), 527-566.

        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): learning rate. Defaults to 1e-2.
            eps (float, optional): finite difference epsilon. Defaults to 1e-2.
            momentum (float, optional): momentum factor. Defaults to 0.
            weight_decay (float, optional): weight decay (L2 penalty). Defaults to 0.
            dampening (float, optional): dampening for momentum. Defaults to 0.
            nesterov (bool, optional): enables Nesterov momentum (supports dampening). Defaults to False.
            formula (_FD_Formulas, optional): finite difference formula. Defaults to "forward".
            n_samples (int, optional): number of random gradient approximations that will be averaged. Defaults to 1.
            distribution (Distributions, optional): distribution for random perturbations. Defaults to "normal".
            randomize_every (int, optional): number of steps between randomizing perturbations. Defaults to 1.
        """
        super().__init__(
            params = params,
            lr = lr,
            eps = eps,
            momentum = momentum,
            weight_decay = weight_decay,
            dampening = dampening,
            nesterov = nesterov,
            formula = formula,
            n_samples = n_samples,
            distribution = distribution,
            randomize_every=randomize_every,
        )

class RandomizedFDMWrapper(ModularOptimizer):
    def __init__(
        self,
        params,
        optimizer: torch.optim.Optimizer,
        eps: float = 1e-3,
        formula: _FD_Formulas = "forward",
        n_samples: int = 1,
        distribution: Distributions = "normal",
        randomize_every: int = 1,
        randomize_closure: bool = False,
    ):
        """Randomized finite difference gradient approximation (e.g. SPSA, RDSA, Nesterov random search).

        With `forward` and `backward` formulas performs `1 + n_samples` evaluations per step;
        with `central` formula performs `2 * n_samples` evaluations per step.

        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups.
            optimizer (torch.optim.Optimizer): optimizer that will perform optimization using RFDM-approximated gradients.
            eps (float, optional): finite difference epsilon. Defaults to 1e-3.
            formula (_FD_Formulas, optional): finite difference formula. Defaults to "forward".
            n_samples (int, optional): number of random gradient approximations that will be averaged. Defaults to 1.
            distribution (Distributions, optional): distribution for random perturbations. Defaults to "normal".
            randomize_every (int, optional): number of steps between randomizing perturbations. Defaults to 1.
            randomize_closure (bool, optional): whether to generate a new random perturbation each time closure
                is evaluated with `backward=True` (this ignores `randomize_every`). Defaults to False. Defaults to False.
        """
        modules = [
            _RandomizedFDM(
                eps=eps,
                formula=formula,
                n_samples=n_samples,
                distribution=distribution,
                randomize_every=randomize_every,
                randomize_closure = randomize_closure,
            ),
            OptimizerWrapper(optimizer, pass_closure=True)
        ]
        super().__init__(params, modules)