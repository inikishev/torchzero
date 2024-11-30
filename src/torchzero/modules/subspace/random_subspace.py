import typing as T
from abc import ABC, abstractmethod
from collections import abc

import torch

from ... import tl
from ...core import OptimizationState, OptimizerModule

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
        self.n_masks = n_pairs
        self.n = n_pairs * 2

    def sample(self, params: tl.TensorList, state: OptimizationState):
        projections = []
        for i in range(self.n_masks):
            mask = params.bernoulli_like(0.5)
            mask2 = 1 - mask
            projections.append(mask)
            projections.append(mask2)

        return projections


class ProjAscent(Projection):
    """Use ascent direction as the projection."""
    def sample(self, params: tl.TensorList, state: OptimizationState):
        if state.ascent_direction is None: raise ValueError
        return [state.ascent_direction]

class ProjAscentRay(Projection):
    def __init__(self, eps = 0.1, n = 1, distribution: tl.Distributions = 'normal', ):
        self.eps = eps
        self.distribution: tl.Distributions = distribution
        self.n = n

    def sample(self, params: tl.TensorList, state: OptimizationState):
        if state.ascent_direction is None: raise ValueError
        mean = params.total_mean().detach().cpu().item()
        return [state.ascent_direction + state.ascent_direction.sample_like(mean * self.eps, distribution=self.distribution) for _ in range(self.n)]

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
            return [state.ascent_direction / state.ascent_direction.total_vector_norm(2) - grad / grad.total_vector_norm(2)] # type:ignore
        else:
            return [state.ascent_direction - grad] # type:ignore

class ProjLastGradDifference(Projection):
    def __init__(self):
        """Use difference between last two gradients as the projection."""
        self.last_grad = None
    def sample(self, params: tl.TensorList, state: OptimizationState):
        if self.last_grad is None:
            self.last_grad = state.maybe_compute_grad_(params)
            return [self.last_grad]
        else:
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
            self.last_direction: tl.TensorList = state.ascent_direction # type:ignore
            return [self.last_direction]
        else:
            diff = state.ascent_direction - self.last_direction # type:ignore
            self.last_direction = state.ascent_direction # type:ignore
            return [diff]

class ProjNormalize(Projection):
    def __init__(self, *projections: Projection):
        """Normalizes all projections to have norm = 1."""
        self.projections = projections

    def sample(self, params: tl.TensorList, state: OptimizationState): # type:ignore
        vecs = [proj for obj in self.projections for proj in obj.sample(params, state)]
        return [v/v.total_vector_norm(2) for v in vecs]

class Subspace(OptimizerModule):
    def __init__(self, projections: Projection | abc.Iterable[Projection], randomize_every: int = 1, ):
        """Optimizes parameters projected into a lower (or higher) dimensional subspace.

        Args:
            projections (Projection | Iterable[Projection]): list of projections.
            randomize_every (int, optional): generates new random projections every n steps. Defaults to 1.
        """
        super().__init__({})
        if isinstance(projections, Projection): projections = [projections]
        self.projections = list(projections)
        self.randomize_every = randomize_every
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
        if self.child is None: raise ValueError('RandomProjection needs a child')
        if state.closure is None: raise ValueError('RandomProjection needs a closure')
        closure = state.closure
        params = self.get_params()

        # every `regenerate_every` steps we generate new random projections.
        if self.current_step % self.randomize_every == 0:

            # generate n projection vetors
            self.projection_vectors = [sample for proj in self.projections for sample in proj.sample(params, state)]

            # child params is n scalars corresponding to each projection vector
            self.projected_params = self.child._params[0] # type:ignore

        # closure that takes the projected params from the child, puts them into full space params, and evaluates the loss
        def projected_closure(backward = True):
            residual = sum([vec * p for vec, p in zip(self.projection_vectors, self.projected_params)])

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
        state.closure = projected_closure
        state.ascent_direction = None
        if state.grad is not None:
            state.grad = tl.TensorList([torch.cat([(params.grad * vec).total_sum().unsqueeze(0) for vec in self.projection_vectors])])
        loss = self.child.step(state)

        # that is going to update child's paramers, which we now project back to the full parameter space
        residual = sum([vec * p for vec, p in zip(self.projection_vectors, self.projected_params)])
        params.add_(residual)

        # projected_params are residuals that have been applied to actual params on previous step in some way
        # therefore they need to now become zero (otherwise they work like momentum with no decay).
        self.projected_params.zero_()

        self.current_step += 1
        return loss