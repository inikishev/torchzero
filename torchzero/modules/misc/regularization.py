import torch

from ...core import Chainable, Module,  Transform
from ...core.reformulation import Reformulation
from ...utils import Distributions, Metrics, NumberList, TensorList, evaluate_metric


class Dropout(Transform):
    """Applies dropout to the update.

    For each weight the update to that weight has ``p`` probability to be set to 0.
    This can be used to implement gradient dropout or update dropout depending on placement.

    Args:
        p (float, optional): probability that update for a weight is replaced with 0. Defaults to 0.5.
        graft (bool, optional):
            if True, update after dropout is rescaled to have the same norm as before dropout. Defaults to False.
        target (Target, optional): what to set on var, refer to documentation. Defaults to 'update'.


    ### Examples:

    Gradient dropout.

    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Dropout(0.5),
        tz.m.Adam(),
        tz.m.LR(1e-3)
    )
    ```

    Update dropout.

    ``python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Adam(),
        tz.m.Dropout(0.5),
        tz.m.LR(1e-3)
    )
    ```

    """
    def __init__(self, p: float = 0.5, graft: bool=False):
        defaults = dict(p=p, graft=graft)
        super().__init__(defaults)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        tensors = TensorList(tensors)
        p = NumberList(s['p'] for s in settings)
        graft = settings[0]['graft']

        if graft:
            target_norm = tensors.global_vector_norm()
            tensors.mul_(tensors.rademacher_like(1-p).add_(1).div_(2))
            return tensors.mul_(target_norm / tensors.global_vector_norm()) # graft

        return tensors.mul_(tensors.rademacher_like(1-p).add_(1).div_(2))

def _bernoulli_like(tensor, p = 0.5, generator = None):
    """p is probability of a 1, other values will be 0."""
    return torch.bernoulli(torch.full_like(tensor, p), generator = generator)

class WeightDropout(Module):
    """
    Changes the closure so that it evaluates loss and gradients with random weights replaced with 0.

    Dropout can be disabled for a parameter by setting ``use_dropout=False`` in corresponding parameter group.

    Args:
        p (float, optional): probability that any weight is replaced with 0. Defaults to 0.5.
    """
    def __init__(self, p: float = 0.5):
        defaults = dict(p=p, use_dropout=True)
        super().__init__(defaults)

    @torch.no_grad
    def update(self, objective):
        closure = objective.closure
        if closure is None: raise RuntimeError('WeightDropout requires closure')
        params = TensorList(objective.params)
        p = NumberList(self.settings[p]['p'] for p in params)

        # create masks
        mask = []
        for p in params:
            prob = self.settings[p]['p']
            use_dropout = self.settings[p]['use_dropout']
            if use_dropout: mask.append(_bernoulli_like(p, prob))
            else: mask.append(torch.ones_like(p))

        # create a closure that evaluates masked parameters
        @torch.no_grad
        def dropout_closure(backward=True):
            orig_params = params.clone()
            params.mul_(mask)
            if backward:
                with torch.enable_grad(): loss = closure()
            else:
                loss = closure(False)
            params.copy_(orig_params)
            return loss

        objective.closure = dropout_closure


class PerturbWeights(Module):
    """
    Changes the closure so that it evaluates loss and gradients at weights perturbed by a random perturbation.

    Can be disabled for a parameter by setting ``perturb=False`` in corresponding parameter group.

    Args:
        alpha (float, optional): multiplier for perturbation magnitude. Defaults to 0.1.
        relative (bool, optional): whether to multiply perturbation by mean absolute value of the parameter. Defaults to True.
        distribution (bool, optional):
            distribution of the random perturbation. Defaults to False.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        relative: bool = True,
        distribution: Distributions = "normal",
        metric: Metrics = "mad",
    ):
        defaults = dict(alpha=alpha, relative=relative, distribution=distribution, metric=metric, perturb=True)
        super().__init__(defaults)

    @torch.no_grad
    def update(self, objective):
        closure = objective.closure
        if closure is None: raise RuntimeError('WeightDropout requires closure')
        params = TensorList(objective.params)

        # create perturbations
        perts = []
        for p in params:
            settings = self.settings[p]
            if not settings['perturb']:
                perts.append(torch.zeros_like(p))
                continue

            alpha = settings['alpha']
            if settings['relative']:
                alpha *= evaluate_metric(p, settings["metric"])

            distribution = self.settings[p]['distribution'].lower()
            if distribution in ('normal', 'gaussian'):
                perts.append(torch.randn_like(p).mul_(alpha))
            elif distribution == 'uniform':
                perts.append(torch.empty_like(p).uniform_(-alpha,alpha))
            elif distribution == 'sphere':
                r = torch.randn_like(p)
                perts.append((r * alpha) / torch.linalg.vector_norm(r)) # pylint:disable=not-callable
            else:
                raise ValueError(distribution)

        @torch.no_grad
        def perturbed_closure(backward=True):
            params.add_(perts)
            if backward:
                with torch.enable_grad(): loss = closure()
            else:
                loss = closure(False)
            params.sub_(perts)
            return loss

        objective.closure = perturbed_closure
