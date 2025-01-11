import typing as T
from abc import ABC, abstractmethod
from collections import abc

import torch

from ... import tensorlist as tl
from ...core import OptimizationState, OptimizerModule, Chain
# this whole thing can also be implemented via parameter vectors.
# Need to test which one is more efficient...

class Projection(ABC):
    n = 1
    @abstractmethod
    def sample(self, params: tl.TensorList, state: OptimizationState) -> list[tl.TensorList]:
        """Generate a projection.

        Args:
            params (tl.TensorList): tensor list of parameters.
            state (OptimizationState): optimization state object.

        Returns:
            projection.
        """

class ProjRandom(Projection):
    def __init__(self, n = 1, distribution: tl.Distributions = 'normal', ):
        self.distribution: tl.Distributions = distribution
        self.n = n

    def sample(self, params: tl.TensorList, state: OptimizationState):
        return [params.sample_like(distribution=self.distribution) for _ in range(self.n)]


class Proj2Masks(Projection):
    def __init__(self, n_pairs = 1):
        """Similar to ProjRandom, but generates pairs of two random masks of 0s and 1s,
        where second mask is an inverse of the first mask."""
        self.n_pairs = n_pairs

    @property
    def n(self):
        return self.n_pairs * 2

    def sample(self, params: tl.TensorList, state: OptimizationState):
        projections = []
        for i in range(self.n_pairs):
            mask = params.bernoulli_like(0.5)
            mask2 = 1 - mask
            projections.append(mask)
            projections.append(mask2)

        return projections


class ProjAscent(Projection):
    """Use ascent direction as the projection."""
    def sample(self, params: tl.TensorList, state: OptimizationState):
        if state.ascent is None: raise ValueError
        return [state.ascent]

class ProjAscentRay(Projection):
    def __init__(self, eps = 0.1, n = 1, distribution: tl.Distributions = 'normal', ):
        self.eps = eps
        self.distribution: tl.Distributions = distribution
        self.n = n

    def sample(self, params: tl.TensorList, state: OptimizationState):
        if state.ascent is None: raise ValueError
        mean = params.total_mean().detach().cpu().item()
        return [state.ascent + state.ascent.sample_like(mean * self.eps, distribution=self.distribution) for _ in range(self.n)]

class ProjGrad(Projection):
    def sample(self, params: tl.TensorList, state: OptimizationState):
        grad = state.maybe_compute_grad_(params)
        return [grad]

class ProjGradRay(Projection):
    def __init__(self, eps = 0.1, n = 1, distribution: tl.Distributions = 'normal', ):
        self.eps = eps
        self.distribution: tl.Distributions = distribution
        self.n = n

    def sample(self, params: tl.TensorList, state: OptimizationState):
        grad = state.maybe_compute_grad_(params)
        mean = params.total_mean().detach().cpu().item()
        return [grad + grad.sample_like(mean * self.eps, distribution=self.distribution) for _ in range(self.n)]

class ProjGradAscentDifference(Projection):
    def __init__(self, normalize=False):
        """Use difference between gradient and ascent direction as projection.

        Args:
            normalize (bool, optional): normalizes grads and ascent projection to have norm = 1. Defaults to False.
        """
        self.normalize = normalize

    def sample(self, params: tl.TensorList, state: OptimizationState):
        grad = state.maybe_compute_grad_(params)
        if self.normalize:
            return [state.ascent / state.ascent.total_vector_norm(2) - grad / grad.total_vector_norm(2)] # type:ignore

        return [state.ascent - grad] # type:ignore

class ProjLastGradDifference(Projection):
    def __init__(self):
        """Use difference between last two gradients as the projection."""
        self.last_grad = None
    def sample(self, params: tl.TensorList, state: OptimizationState):
        if self.last_grad is None:
            self.last_grad = state.maybe_compute_grad_(params)
            return [self.last_grad]

        grad = state.maybe_compute_grad_(params)
        diff = grad - self.last_grad
        self.last_grad = grad
        return [diff]

class ProjLastAscentDifference(Projection):
    def __init__(self):
        """Use difference between last two ascent directions as the projection."""
        self.last_direction = T.cast(tl.TensorList, None)

    def sample(self, params: tl.TensorList, state: OptimizationState):
        if self.last_direction is None:
            self.last_direction: tl.TensorList = state.ascent # type:ignore
            return [self.last_direction]

        diff = state.ascent - self.last_direction # type:ignore
        self.last_direction = state.ascent # type:ignore
        return [diff]

class ProjNormalize(Projection):
    def __init__(self, *projections: Projection):
        """Normalizes all projections to have norm = 1."""
        self.projections = projections

    @property
    def n(self):
        return sum(proj.n for proj in self.projections)

    def sample(self, params: tl.TensorList, state: OptimizationState): # type:ignore
        vecs = [proj for obj in self.projections for proj in obj.sample(params, state)]
        norms = [v.total_vector_norm(2) for v in vecs]
        return [v/norm if norm!=0 else v.randn_like() for v,norm in zip(vecs,norms)]

class Subspace(OptimizerModule):
    """This is pretty inefficient, I thought of a much better way to do this via jvp and I will rewrite this soon.

    Optimizes parameters projected into a lower (or higher) dimensional subspace.

    The subspace is a bunch of projections that go through the current point. Projections can be random,
    or face in the direction of the gradient, or difference between last two gradients, etc. The projections
    are updated every `update_every` steps.

    Notes:
        This doesn't work with anything that directly calculates the hessian or other quantities via `torch.autograd.grad`,
        like `ExactNewton`. I will have to manually implement a subspace version for it.

        This also zeroes parameters after each step, meaning it won't work with some integrations like nevergrad
        (as they store their own parameters which don't get zeroed). It does however work with integrations like
        `scipy.optimize` because they performs a full minimization on each step.
    Another version of this which doesn't zero the params is under way.

    Args:
        projections (Projection | Iterable[Projection]):
            list of projections - `Projection` objects that define the directions of the projections.
            Each Projection object may generate one or multiple directions.
        update_every (int, optional): generates new projections every n steps. Defaults to 1.
    """
    def __init__(
        self,
        modules: OptimizerModule | abc.Iterable[OptimizerModule],
        projections: Projection | abc.Iterable[Projection],
        update_every: int | None = 1,
    ):
        super().__init__({})
        if isinstance(projections, Projection): projections = [projections]
        self.projections = list(projections)
        self._set_child_('subspace', modules)
        self.update_every = update_every
        self.current_step = 0

        # cast them because they are guaranteed to be assigned on 1st step.
        self.projection_vectors = T.cast(list[tl.TensorList], None)
        self.projected_params = T.cast(torch.Tensor, None)


    def _update_child_params_(self, child: "OptimizerModule"):
        dtype = self._params[0].dtype
        device = self._params[0].device
        params = [torch.zeros(sum(proj.n for proj in self.projections), dtype = dtype, device = device, requires_grad=True)]
        if not child._initialized:
            child._initialize_(params)
        else:
            child.param_groups = []
            child.add_param_group({"params": params})

    @torch.no_grad
    def step(self, state):
        #if self.next_module is None: raise ValueError('RandomProjection needs a child')
        if state.closure is None: raise ValueError('RandomProjection needs a closure')
        closure = state.closure
        params = self.get_params()

        # every `regenerate_every` steps we generate new random projections.
        if self.current_step == 0 or (self.update_every is not None and self.current_step % self.update_every == 0):

            # generate n projection vetors
            self.projection_vectors = [sample for proj in self.projections for sample in proj.sample(params, state)]

            # child params is n scalars corresponding to each projection vector
            self.projected_params = self.children['subspace']._params[0] # type:ignore

        # closure that takes the projected params from the child, puts them into full space params, and evaluates the loss
        def projected_closure(backward = True):
            residual = sum(vec * p for vec, p in zip(self.projection_vectors, self.projected_params))

            # this in-place operation prevents autodiff from working
            # we manually calculate the gradients as they are just a product
            # therefore we need torch.no_grad here because optimizers call closure under torch.enabled_grad
            with torch.no_grad(): params.add_(residual)

            loss = closure(backward)

            if backward:
                self.projected_params.grad = torch.cat([(params.grad * vec).total_sum().unsqueeze(0) for vec in self.projection_vectors])
            with torch.no_grad(): params.sub_(residual)
            return loss

        # # if ascent direction is provided,
        # # project the ascent direction into the projection space (need to test if this works)
        # if ascent_direction is not None:
        #     ascent_direction = tl.sum([ascent_direction*v for v in self.projection_vectors])

        # perform a step with the child
        subspace_state = state.copy(False)
        subspace_state.closure = projected_closure
        subspace_state.ascent = None
        if subspace_state.grad is not None:
            subspace_state.grad = tl.TensorList([torch.cat([(params.grad * vec).total_sum().unsqueeze(0) for vec in self.projection_vectors])])
        self.children['subspace'].step(subspace_state) # type:ignore

        # that is going to update child's paramers, which we now project back to the full parameter space
        residual = tl.sum([vec * p for vec, p in zip(self.projection_vectors, self.projected_params)])
        state.ascent = residual.neg_()

        # move fx0 and fx0 approx to state
        if subspace_state.fx0 is not None: state.fx0 = subspace_state.fx0
        if subspace_state.fx0_approx is not None: state.fx0 = subspace_state.fx0_approx
        # projected_params are residuals that have been applied to actual params on previous step in some way
        # therefore they need to now become zero (otherwise they work like momentum with no decay).
        # note: THIS WON'T WORK WITH INTEGRATIONS, UNLESS THEY PERFORM FULL MINIMIZATION EACH STEP
        # because their params won't be zeroed.
        self.projected_params.zero_()

        self.current_step += 1
        return self._update_params_or_step_with_next(state)