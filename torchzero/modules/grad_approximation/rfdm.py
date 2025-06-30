from collections.abc import Callable
from typing import Any
from functools import partial
import torch

from ...utils import TensorList, Distributions, NumberList
from .grad_approximator import GradApproximator, GradTarget, _FD_Formula

def _rforward2(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
    """p_fn is a function that returns the perturbation.
    It may return pre-generated one or generate one deterministically from a seed as in MeZO.
    Returned perturbation must be multiplied by `h`."""
    if f_0 is None: f_0 = closure(False)
    params += p_fn()
    f_1 = closure(False)
    params -= p_fn()
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_0, (f_1 - f_0) / h # (loss, loss_approx, grad)

def _rbackward2(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
    if f_0 is None: f_0 = closure(False)
    params -= p_fn()
    f_m1 = closure(False)
    params += p_fn()
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_0, (f_0 - f_m1) / h

def _rcentral2(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: Any):
    params += p_fn()
    f_1 = closure(False)

    params -= p_fn() * 2
    f_m1 = closure(False)

    params += p_fn()
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_1, (f_1 - f_m1) / (2 * h)

def _rforward3(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
    if f_0 is None: f_0 = closure(False)
    params += p_fn()
    f_1 = closure(False)

    params += p_fn()
    f_2 = closure(False)

    params -= p_fn() * 2
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_0, (-3*f_0 + 4*f_1 - f_2) / (2 * h)

def _rbackward3(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
    if f_0 is None: f_0 = closure(False)

    params -= p_fn()
    f_m1 = closure(False)

    params -= p_fn()
    f_m2 = closure(False)

    params += p_fn() * 2
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_0, (f_m2 - 4*f_m1 + 3*f_0) / (2 * h)

def _rcentral4(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
    params += p_fn()
    f_1 = closure(False)

    params += p_fn()
    f_2 = closure(False)

    params -= p_fn() * 3
    f_m1 = closure(False)

    params -= p_fn()
    f_m2 = closure(False)

    params += p_fn() * 2
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_1, (f_m2 - 8*f_m1 + 8*f_1 - f_2) / (12 * h)

# some good ones
# Pachalyl S. et al. Generalized simultaneous perturbation-based gradient search with reduced estimator bias //IEEE Transactions on Automatic Control. – 2025.
# Three measurements GSPSA is _rforward3
# Four measurements GSPSA
def _rforward4(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
    if f_0 is None: f_0 = closure(False)
    params += p_fn()
    f_1 = closure(False)

    params += p_fn()
    f_2 = closure(False)

    params += p_fn()
    f_3 = closure(False)

    params -= p_fn() * 3
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_0, (2*f_3 - 9*f_2 + 18*f_1 - 11*f_0) / (6 * h)

def _rforward5(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
    if f_0 is None: f_0 = closure(False)
    params += p_fn()
    f_1 = closure(False)

    params += p_fn()
    f_2 = closure(False)

    params += p_fn()
    f_3 = closure(False)

    params += p_fn()
    f_4 = closure(False)

    params -= p_fn() * 4
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_0, (-3*f_4 + 16*f_3 - 36*f_2 + 48*f_1 - 25*f_0) / (12 * h)

# another central4
def _bgspsa4(closure: Callable[..., float], params:TensorList, p_fn:Callable[[], TensorList], h, f_0: float | None):
    params += p_fn()
    f_1 = closure(False)

    params += p_fn() * 2
    f_3 = closure(False)

    params -= p_fn() * 4
    f_m1 = closure(False)

    params -= p_fn() * 2
    f_m3 = closure(False)

    params += p_fn() * 3
    h = h**2 # because perturbation already multiplied by h
    return f_0, f_1, (27*f_1 - f_m1 - f_3 + f_m3) / (48 * h)


_RFD_FUNCS = {
    "forward": _rforward2,
    "forward2": _rforward2,
    "backward": _rbackward2,
    "backward2": _rbackward2,
    "central": _rcentral2,
    "central2": _rcentral2,
    "central3": _rcentral2,
    "forward3": _rforward3,
    "backward3": _rbackward3,
    "central4": _rcentral4,
    "forward4": _rforward4,
    "forward5": _rforward5,
    "bspsa4": _bgspsa4,
}


class RandomizedFDM(GradApproximator):
    """Gradient approximation via a randomized finite-difference method.

    .. note::
        This module is a gradient approximator. It modifies the closure to evaluate the estimated gradients,
        and further closure-based modules will use the modified closure. All modules after this will use estimated gradients.

    Args:
        h (float, optional): finite difference step size of jvp_method is set to `forward` or `central`. Defaults to 1e-3.
        n_samples (int, optional): number of random gradient samples. Defaults to 1.
        formula (_FD_Formula, optional): finite difference formula. Defaults to 'central2'.
        distribution (Distributions, optional): distribution. Defaults to "rademacher".
            If this is set to a value higher than zero, instead of using directional derivatives in a new random direction on each step, the direction changes gradually with momentum based on this value. This may make it possible to use methods with memory. Defaults to 0.
        beta (float, optional): optinal momentum for generated perturbations. Defaults to 1e-3.
        pre_generate (bool, optional):
            whether to pre-generate gradient samples before each step. If samples are not pre-generated, whenever a method performs multiple closure evaluations, the gradient will be evaluated in different directions each time. Defaults to True.
        seed (int | None | torch.Generator, optional): Seed for random generator. Defaults to None.
        target (GradTarget, optional): what to set on var. Defaults to "closure".

    Examples:
        #### Simultaneous perturbation stochastic approximation (SPSA) method

        SPSA is randomized finite differnce with rademacher distribution and central formula.

        .. code-block:: python

            spsa = tz.Modular(
                model.parameters(),
                tz.m.RandomizedFDM(formula="central", distribution="rademacher"),
                tz.m.LR(1e-2)
            )

        #### Random-direction stochastic approximation (RDSA) method

        RDSA is randomized finite differnce with usually gaussian distribution and central formula.

        .. code-block:: python

            rdsa = tz.Modular(
                model.parameters(),
                tz.m.RandomizedFDM(formula="central", distribution="gaussian"),
                tz.m.LR(1e-2)
            )

        #### RandomizedFDM with momentum

        Momentum might help by reducing the variance of the estimated gradients.

        .. code-block:: python

            momentum_spsa = tz.Modular(
                model.parameters(),
                tz.m.RandomizedFDM(),
                tz.m.HeavyBall(0.9),
                tz.m.LR(1e-3)
            )

        #### Gaussian smoothing method

        GS uses many gaussian samples with possibly a larger finite difference step size.

        .. code-block:: python

            gs = tz.Modular(
                model.parameters(),
                tz.m.RandomizedFDM(n_samples=100, distribution="gaussian", formula="forward2", h=1e-1),
                tz.m.NewtonCG(hvp_method="forward"),
                tz.m.Backtracking()
            )

        #### SPSA-NewtonCG

        NewtonCG with hessian-vector product estimated via gradient difference
        calls closure multiple times per step. If each closure call estimates gradients
        with different perturbations, NewtonCG is unable to produce useful directions.

        By setting pre_generate to True, perturbations are generated once before each step,
        and each closure call estimates gradients using the same pre-generated perturbations.
        This way closure-based algorithms are able to use gradients estimated in a consistent way.

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.RandomizedFDM(n_samples=10),
                tz.m.NewtonCG(hvp_method="forward", pre_generate=True),
                tz.m.Backtracking()
            )

        #### SPSA-BFGS

        L-BFGS uses a memory of past parameter and gradient differences. If past gradients
        were estimated with different perturbations, L-BFGS directions will be useless.

        To alleviate this momentum can be added to random perturbations to make sure they only
        change by a little bit, and the history stays relevant. The momentum is determined by the :code:`beta` parameter.
        The disadvantage is that the subspace the algorithm is able to explore changes slowly.

        Additionally we will reset BFGS memory every 100 steps to remove influence from old gradient estimates.

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.RandomizedFDM(n_samples=10, pre_generate=True, beta=0.99),
                tz.m.BFGS(reset_interval=100),
                tz.m.Backtracking()
            )
    """
    PRE_MULTIPLY_BY_H = True
    def __init__(
        self,
        h: float = 1e-3,
        n_samples: int = 1,
        formula: _FD_Formula = "central",
        distribution: Distributions = "rademacher",
        beta: float = 0,
        pre_generate = True,
        seed: int | None | torch.Generator = None,
        target: GradTarget = "closure",
    ):
        defaults = dict(h=h, formula=formula, n_samples=n_samples, distribution=distribution, beta=beta, pre_generate=pre_generate, seed=seed)
        super().__init__(defaults, target=target)

    def reset(self):
        self.state.clear()
        generator = self.global_state.get('generator', None) # avoid resetting generator
        self.global_state.clear()
        if generator is not None: self.global_state['generator'] = generator

    def _get_generator(self, seed: int | None | torch.Generator, params: list[torch.Tensor]):
        if 'generator' not in self.global_state:
            if isinstance(seed, torch.Generator): self.global_state['generator'] = seed
            elif seed is not None: self.global_state['generator'] = torch.Generator(params[0].device).manual_seed(seed)
            else: self.global_state['generator'] = None
        return self.global_state['generator']

    def pre_step(self, var):
        h, beta = self.get_settings(var.params, 'h', 'beta')
        settings = self.settings[var.params[0]]
        n_samples = settings['n_samples']
        distribution = settings['distribution']
        pre_generate = settings['pre_generate']

        if pre_generate:
            params = TensorList(var.params)
            generator = self._get_generator(settings['seed'], var.params)
            perturbations = [params.sample_like(distribution=distribution, generator=generator) for _ in range(n_samples)]

            if self.PRE_MULTIPLY_BY_H:
                torch._foreach_mul_([p for l in perturbations for p in l], [v for vv in h for v in [vv]*n_samples])

            if all(i==0 for i in beta):
                # just use pre-generated perturbations
                for param, prt in zip(params, zip(*perturbations)):
                    self.state[param]['perturbations'] = prt

            else:
                # lerp old and new perturbations. This makes the subspace change gradually
                # which in theory might improve algorithms with history
                for i,p in enumerate(params):
                    state = self.state[p]
                    if 'perturbations' not in state: state['perturbations'] = [p[i] for p in perturbations]

                cur = [self.state[p]['perturbations'][:n_samples] for p in params]
                cur_flat = [p for l in cur for p in l]
                new_flat = [p for l in zip(*perturbations) for p in l]
                betas = [1-v for b in beta for v in [b]*n_samples]
                torch._foreach_lerp_(cur_flat, new_flat, betas)

    @torch.no_grad
    def approximate(self, closure, params, loss):
        params = TensorList(params)
        orig_params = params.clone() # store to avoid small changes due to float imprecision
        loss_approx = None

        h = NumberList(self.settings[p]['h'] for p in params)
        settings = self.settings[params[0]]
        n_samples = settings['n_samples']
        fd_fn = _RFD_FUNCS[settings['formula']]
        default = [None]*n_samples
        perturbations = list(zip(*(self.state[p].get('perturbations', default) for p in params)))
        distribution = settings['distribution']
        generator = self._get_generator(settings['seed'], params)

        grad = None
        for i in range(n_samples):
            prt = perturbations[i]
            if prt[0] is None: prt = params.sample_like(distribution=distribution, generator=generator).mul_(h)
            else: prt = TensorList(prt)

            loss, loss_approx, d = fd_fn(closure=closure, params=params, p_fn=lambda: prt, h=h, f_0=loss)
            if grad is None: grad = prt * d
            else: grad += prt * d

        params.set_(orig_params)
        assert grad is not None
        if n_samples > 1: grad.div_(n_samples)
        return grad, loss, loss_approx

class SPSA(RandomizedFDM):
    """
    Gradient approximation via Simultaneous perturbation stochastic approximation (SPSA) method.

    .. note::
        This module is a gradient approximator. It modifies the closure to evaluate the estimated gradients,
        and further closure-based modules will use the modified closure. All modules after this will use estimated gradients.


    Args:
        h (float, optional): finite difference step size of jvp_method is set to `forward` or `central`. Defaults to 1e-3.
        n_samples (int, optional): number of random gradient samples. Defaults to 1.
        formula (_FD_Formula, optional): finite difference formula. Defaults to 'central2'.
        distribution (Distributions, optional): distribution. Defaults to "rademacher".
        beta (float, optional):
            If this is set to a value higher than zero, instead of using directional derivatives in a new random direction on each step, the direction changes gradually with momentum based on this value. This may make it possible to use methods with memory. Defaults to 0.
        pre_generate (bool, optional):
            whether to pre-generate gradient samples before each step. If samples are not pre-generated, whenever a method performs multiple closure evaluations, the gradient will be evaluated in different directions each time. Defaults to True.
        seed (int | None | torch.Generator, optional): Seed for random generator. Defaults to None.
        target (GradTarget, optional): what to set on var. Defaults to "closure".

    References:
        Chen, Y. (2021). Theoretical study and comparison of SPSA and RDSA algorithms. arXiv preprint arXiv:2107.12771. https://arxiv.org/abs/2107.12771
    """

class RDSA(RandomizedFDM):
    """
    Gradient approximation via Random-direction stochastic approximation (RDSA) method.

    .. note::
        This module is a gradient approximator. It modifies the closure to evaluate the estimated gradients,
        and further closure-based modules will use the modified closure. All modules after this will use estimated gradients.

    Args:
        h (float, optional): finite difference step size of jvp_method is set to `forward` or `central`. Defaults to 1e-3.
        n_samples (int, optional): number of random gradient samples. Defaults to 1.
        formula (_FD_Formula, optional): finite difference formula. Defaults to 'central2'.
        distribution (Distributions, optional): distribution. Defaults to "gaussian".
        beta (float, optional):
            If this is set to a value higher than zero, instead of using directional derivatives in a new random direction on each step, the direction changes gradually with momentum based on this value. This may make it possible to use methods with memory. Defaults to 0.
        pre_generate (bool, optional):
            whether to pre-generate gradient samples before each step. If samples are not pre-generated, whenever a method performs multiple closure evaluations, the gradient will be evaluated in different directions each time. Defaults to True.
        seed (int | None | torch.Generator, optional): Seed for random generator. Defaults to None.
        target (GradTarget, optional): what to set on var. Defaults to "closure".

    References:
        Chen, Y. (2021). Theoretical study and comparison of SPSA and RDSA algorithms. arXiv preprint arXiv:2107.12771. https://arxiv.org/abs/2107.12771

    """
    def __init__(
        self,
        h: float = 1e-3,
        n_samples: int = 1,
        formula: _FD_Formula = "central2",
        distribution: Distributions = "gaussian",
        beta: float = 0,
        pre_generate = True,
        target: GradTarget = "closure",
        seed: int | None | torch.Generator = None,
    ):
        super().__init__(h=h, n_samples=n_samples,formula=formula,distribution=distribution,beta=beta,pre_generate=pre_generate,target=target,seed=seed)

class GaussianSmoothing(RandomizedFDM):
    """
    Gradient approximation via Gaussian smoothing method.

    .. note::
        This module is a gradient approximator. It modifies the closure to evaluate the estimated gradients,
        and further closure-based modules will use the modified closure. All modules after this will use estimated gradients.

    Args:
        h (float, optional): finite difference step size of jvp_method is set to `forward` or `central`. Defaults to 1e-2.
        n_samples (int, optional): number of random gradient samples. Defaults to 100.
        formula (_FD_Formula, optional): finite difference formula. Defaults to 'forward2'.
        distribution (Distributions, optional): distribution. Defaults to "gaussian".
        beta (float, optional):
            If this is set to a value higher than zero, instead of using directional derivatives in a new random direction on each step, the direction changes gradually with momentum based on this value. This may make it possible to use methods with memory. Defaults to 0.
        pre_generate (bool, optional):
            whether to pre-generate gradient samples before each step. If samples are not pre-generated, whenever a method performs multiple closure evaluations, the gradient will be evaluated in different directions each time. Defaults to True.
        seed (int | None | torch.Generator, optional): Seed for random generator. Defaults to None.
        target (GradTarget, optional): what to set on var. Defaults to "closure".


    References:
        Yurii Nesterov, Vladimir Spokoiny. (2015). Random Gradient-Free Minimization of Convex Functions. https://gwern.net/doc/math/2015-nesterov.pdf
    """
    def __init__(
        self,
        h: float = 1e-2,
        n_samples: int = 100,
        formula: _FD_Formula = "forward2",
        distribution: Distributions = "gaussian",
        beta: float = 0,
        pre_generate = True,
        target: GradTarget = "closure",
        seed: int | None | torch.Generator = None,
    ):
        super().__init__(h=h, n_samples=n_samples,formula=formula,distribution=distribution,beta=beta,pre_generate=pre_generate,target=target,seed=seed)

class MeZO(GradApproximator):
    """Gradient approximation via memory-efficient zeroth order optimizer (MeZO) - https://arxiv.org/abs/2305.17333.

    .. note::
        This module is a gradient approximator. It modifies the closure to evaluate the estimated gradients,
        and further closure-based modules will use the modified closure. All modules after this will use estimated gradients.

    Args:
        h (float, optional): finite difference step size of jvp_method is set to `forward` or `central`. Defaults to 1e-3.
        n_samples (int, optional): number of random gradient samples. Defaults to 1.
        formula (_FD_Formula, optional): finite difference formula. Defaults to 'central2'.
        distribution (Distributions, optional): distribution. Defaults to "rademacher".
            If this is set to a value higher than zero, instead of using directional derivatives in a new random direction on each step, the direction changes gradually with momentum based on this value. This may make it possible to use methods with memory. Defaults to 0.
        target (GradTarget, optional): what to set on var. Defaults to "closure".

    References:
        Malladi, S., Gao, T., Nichani, E., Damian, A., Lee, J. D., Chen, D., & Arora, S. (2023). Fine-tuning language models with just forward passes. Advances in Neural Information Processing Systems, 36, 53038-53075. https://arxiv.org/abs/2305.17333
    """

    def __init__(self, h: float=1e-3, n_samples: int = 1, formula: _FD_Formula = 'central2',
                 distribution: Distributions = 'rademacher', target: GradTarget = 'closure'):

        defaults = dict(h=h, formula=formula, n_samples=n_samples, distribution=distribution)
        super().__init__(defaults, target=target)

    def _seeded_perturbation(self, params: list[torch.Tensor], distribution, seed, h):
        return TensorList(params).sample_like(
            distribution=distribution, generator=torch.Generator(params[0].device).manual_seed(seed)
        ).mul_(h)

    def pre_step(self, var):
        h = NumberList(self.settings[p]['h'] for p in var.params)
        settings = self.settings[var.params[0]]
        n_samples = settings['n_samples']
        distribution = settings['distribution']

        step = var.current_step

        # create functions that generate a deterministic perturbation from seed based on current step
        prt_fns = []
        for i in range(n_samples):

            prt_fn = partial(self._seeded_perturbation, params=var.params, distribution=distribution, seed=1_000_000*step + i, h=h)
            prt_fns.append(prt_fn)

        self.global_state['prt_fns'] = prt_fns

    @torch.no_grad
    def approximate(self, closure, params, loss):
        params = TensorList(params)
        loss_approx = None

        h = NumberList(self.settings[p]['h'] for p in params)
        settings = self.settings[params[0]]
        n_samples = settings['n_samples']
        fd_fn = _RFD_FUNCS[settings['formula']]
        prt_fns = self.global_state['prt_fns']

        grad = None
        for i in range(n_samples):
            loss, loss_approx, d = fd_fn(closure=closure, params=params, p_fn=prt_fns[i], h=h, f_0=loss)
            if grad is None: grad = prt_fns[i]().mul_(d)
            else: grad += prt_fns[i]().mul_(d)

        assert grad is not None
        if n_samples > 1: grad.div_(n_samples)
        return grad, loss, loss_approx