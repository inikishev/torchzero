import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from functools import partial
from typing import Literal, cast

import torch

from ...core import Chainable, Optimizer, Module, Objective
from ...core.reformulation import Reformulation
from ...utils import Distributions, NumberList, TensorList
from ..termination import TerminationCriteriaBase, make_termination_criteria


def _reset_except_self(objective: Objective, modules, self: Module):
    for m in modules:
        if m is not self:
            m.reset()


class GradientSampling(Reformulation):
    """Samples and aggregates gradients and values at perturbed points.

    This module can be used for gaussian homotopy and gradient sampling methods.

    Args:
        modules (Chainable | None, optional):
            modules that will be optimizing the modified objective.
            if None, returns gradient of the modified objective as the update. Defaults to None.
        sigma (float, optional): initial magnitude of the perturbations. Defaults to 1.
        n (int, optional): number of perturbations per step. Defaults to 100.
        aggregate (str, optional):
            how to aggregate values and gradients
            - "mean" - uses mean of the gradients, as in gaussian homotopy.
            - "max" - uses element-wise maximum of the gradients.
            - "min" - uses element-wise minimum of the gradients.
            - "min-norm" - picks gradient with the lowest norm.

            Defaults to 'mean'.
        distribution (Distributions, optional): distribution for random perturbations. Defaults to 'gaussian'.
        include_x0 (bool, optional): whether to include gradient at un-perturbed point. Defaults to True.
        fixed (bool, optional):
            if True, perturbations do not get replaced by new random perturbations until termination criteria is satisfied. Defaults to True.
        pre_generate (bool, optional):
            if True, perturbations are pre-generated before each step.
            This requires more memory to store all of them,
            but ensures they do not change when closure is evaluated multiple times.
            Defaults to True.
        termination (TerminationCriteriaBase | Sequence[TerminationCriteriaBase] | None, optional):
            a termination criteria module, sigma will be multiplied by ``decay`` when termination criteria is satisfied,
            and new perturbations will be generated if ``fixed``. Defaults to None.
        decay (float, optional): sigma multiplier on termination criteria. Defaults to 2/3.
        reset_on_termination (bool, optional): whether to reset states of all other modules on termination. Defaults to True.
        sigma_strategy (str | None, optional):
            strategy for adapting sigma. If condition is satisfied, sigma is multiplied by ``sigma_nplus``,
            otherwise it is multiplied by ``sigma_nminus``.
            - "grad-norm" - at least ``sigma_target`` gradients should have lower norm than at un-perturbed point.
            - "value" - at least ``sigma_target`` values (losses) should be lower than at un-perturbed point.
            - None - doesn't use adaptive sigma.

            This introduces a side-effect to the closure, so it should be left at None of you use
            trust region or line search to optimize the modified objective.
            Defaults to None.
        sigma_target (int, optional):
            number of elements to satisfy the condition in ``sigma_strategy``. Defaults to 1.
        sigma_nplus (float, optional): sigma multiplier when ``sigma_strategy`` condition is satisfied. Defaults to 4/3.
        sigma_nminus (float, optional): sigma multiplier when ``sigma_strategy`` condition is not satisfied. Defaults to 2/3.
        seed (int | None, optional): seed. Defaults to None.
    """
    def __init__(
        self,
        modules: Chainable | None = None,
        sigma: float = 1.,
        n:int = 100,
        aggregate: Literal['mean', 'max', 'min', 'min-norm', 'min-value'] = 'mean',
        distribution: Distributions = 'gaussian',
        include_x0: bool = True,

        fixed: bool=True,
        pre_generate: bool = True,
        termination: TerminationCriteriaBase | Sequence[TerminationCriteriaBase] | None = None,
        decay: float = 2/3,
        reset_on_termination: bool = True,

        sigma_strategy: Literal['grad-norm', 'value'] | None = None,
        sigma_target: int | float = 0.2,
        sigma_nplus: float = 4/3,
        sigma_nminus: float = 2/3,

        seed: int | None = None,
    ):

        defaults = dict(sigma=sigma, n=n, aggregate=aggregate, distribution=distribution, seed=seed, include_x0=include_x0, fixed=fixed, decay=decay, reset_on_termination=reset_on_termination, sigma_strategy=sigma_strategy, sigma_target=sigma_target, sigma_nplus=sigma_nplus, sigma_nminus=sigma_nminus, pre_generate=pre_generate)
        super().__init__(defaults, modules)

        if termination is not None:
            self.set_child('termination', make_termination_criteria(extra=termination))

    @torch.no_grad
    def pre_step(self, objective):
        params = TensorList(objective.params)

        fixed = self.defaults['fixed']

        # check termination criteria
        if 'termination' in self.children:
            termination = cast(TerminationCriteriaBase, self.children['termination'])
            if termination.should_terminate(objective):

                # decay sigmas
                states = [self.state[p] for p in params]
                settings = [self.settings[p] for p in params]

                for state, setting in zip(states, settings):
                    if 'sigma' not in state: state['sigma'] = setting['sigma']
                    state['sigma'] *= setting['decay']

                # reset on sigmas decay
                if self.defaults['reset_on_termination']:
                    objective.post_step_hooks.append(partial(_reset_except_self, self=self))

                # clear perturbations
                self.global_state.pop('perts', None)

        # pre-generate perturbations if not already pre-generated or not fixed
        if self.defaults['pre_generate'] and (('perts' not in self.global_state) or (not fixed)):
            states = [self.state[p] for p in params]
            settings = [self.settings[p] for p in params]

            n = self.defaults['n'] - self.defaults['include_x0']
            generator = self.get_generator(params[0].device, self.defaults['seed'])

            perts = [params.sample_like(self.defaults['distribution'], generator=generator) for _ in range(n)]

            self.global_state['perts'] = perts

    @torch.no_grad
    def closure(self, backward, closure, params, objective):
        params = TensorList(params)
        loss_agg = None
        grad_agg = None

        states = [self.state[p] for p in params]
        settings = [self.settings[p] for p in params]
        sigma_inits = [s['sigma'] for s in settings]
        sigmas = [s.setdefault('sigma', si) for s, si in zip(states, sigma_inits)]

        include_x0 = self.defaults['include_x0']
        pre_generate = self.defaults['pre_generate']
        aggregate: Literal['mean', 'max', 'min', 'min-norm', 'min-value'] = self.defaults['aggregate']
        sigma_strategy: Literal['grad-norm', 'value'] | None = self.defaults['sigma_strategy']
        distribution = self.defaults['distribution']
        generator = self.get_generator(params[0].device, self.defaults['seed'])


        n_finite = 0
        n_good = 0
        f_0 = None; g_0 = None

        # evaluate at x_0
        if include_x0:
            f_0 = objective.get_loss(backward=backward)

            isfinite = math.isfinite(f_0)
            if isfinite:
                n_finite += 1
                loss_agg = f_0

            if backward:
                g_0 = objective.get_grads()
                if isfinite: grad_agg = g_0

        # evaluate at x_0 + p for each perturbation
        if pre_generate:
            perts = self.global_state['perts']
        else:
            perts = [None] * (self.defaults['n'] - include_x0)

        x_0 = [p.clone() for p in params]

        for pert in perts:
            loss = None; grad = None

            # generate if not pre-generated
            if pert is None:
                pert = params.sample_like(distribution, generator=generator)

            # add perturbation and evaluate
            pert = pert * sigmas
            torch._foreach_add_(params, pert)

            with torch.enable_grad() if backward else nullcontext():
                loss = closure(backward)

            if math.isfinite(loss):
                n_finite += 1

                # add loss
                if loss_agg is None:
                    loss_agg = loss
                else:
                    if aggregate == 'mean':
                        loss_agg += loss

                    elif (aggregate=='min') or (aggregate=='min-value') or (aggregate=='min-norm' and not backward):
                        loss_agg = loss_agg.clamp(max=loss)

                    elif aggregate == 'max':
                        loss_agg = loss_agg.clamp(min=loss)

                # add grad
                if backward:
                    grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                    if grad_agg is None:
                        grad_agg = grad
                    else:
                        if aggregate == 'mean':
                            torch._foreach_add_(grad_agg, grad)

                        elif aggregate == 'min':
                            grad_agg_abs = torch._foreach_abs(grad_agg)
                            torch._foreach_minimum_(grad_agg_abs, torch._foreach_abs(grad))
                            grad_agg = [g_abs.copysign(g) for g_abs, g in zip(grad_agg_abs, grad_agg)]

                        elif aggregate == 'max':
                            grad_agg_abs = torch._foreach_abs(grad_agg)
                            torch._foreach_maximum_(grad_agg_abs, torch._foreach_abs(grad))
                            grad_agg = [g_abs.copysign(g) for g_abs, g in zip(grad_agg_abs, grad_agg)]

                        elif aggregate == 'min-norm':
                            if TensorList(grad).global_vector_norm() < TensorList(grad_agg).global_vector_norm():
                                grad_agg = grad
                                loss_agg = loss

                        elif aggregate == 'min-value':
                            if loss < loss_agg:
                                grad_agg = grad
                                loss_agg = loss

            # undo perturbation
            torch._foreach_copy_(params, x_0)

            # adaptive sigma
            # by value
            if sigma_strategy == 'value':
                if f_0 is None:
                    with torch.enable_grad() if backward else nullcontext():
                        f_0 = closure(False)

                if loss < f_0:
                    n_good += 1

            # by gradient norm
            elif sigma_strategy == 'grad-norm' and backward and math.isfinite(loss):
                assert grad is not None
                if g_0 is None:
                    with torch.enable_grad() if backward else nullcontext():
                        closure()
                        g_0 = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

                if TensorList(grad).global_vector_norm() < TensorList(g_0).global_vector_norm():
                    n_good += 1

        # update sigma if strategy is enabled
        if sigma_strategy is not None:

            sigma_target = self.defaults['sigma_target']
            if isinstance(sigma_target, float):
                sigma_target = int(max(1, n_finite * sigma_target))

            if n_good >= sigma_target:
                key = 'sigma_nplus'
            else:
                key = 'sigma_nminus'

            for p in params:
                self.state[p]['sigma'] *= self.settings[p][key]

        # if no finite losses, just return inf
        if n_finite == 0:
            assert loss_agg is None and grad_agg is None
            loss = torch.tensor(torch.inf, dtype=params[0].dtype, device=params[0].device)
            grad = [torch.full_like(p, torch.inf) for p in params]
            return loss, grad

        assert loss_agg is not None

        # no post processing needed when aggregate is 'max', 'min', 'min-norm', 'min-value'
        if aggregate != 'mean':
            return loss_agg, grad_agg

        # on mean divide by number of evals
        loss_agg /= n_finite

        if backward:
            assert grad_agg is not None
            torch._foreach_div_(grad_agg, n_finite)

        return loss_agg, grad_agg