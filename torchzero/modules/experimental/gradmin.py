import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

import torch

from ...core import Module, Var
from ...utils import NumberList, TensorList
from ...utils.derivatives import jacobian_wrt
from ..grad_approximation import GradApproximator, GradTarget
from ..smoothing.gaussian import Reformulation



class GradMin(Reformulation):
    """Reformulates the objective to minimize sum of gradient magnitudes via autograd. This is not expected to be practical.

    Args:
        loss_term (float, optional): adds loss value times this to sum of gradient magnitudes. Defaults to 1.
        relative (bool, optional): whether to make loss_term relative to gradient magnitude. Defaults to False.
        graft (bool, optional): whether to make loss term same as gradient magnitude. Defaults to False.
        square (bool, optional): whether to use sum of squared gradient magnitudes, if False uses absolute values. Defaults to False.
        mean (bool, optional): whether to use mean, if False uses sum. Defaults to True.
        maximize_grad (bool, optional): whether to maximize gradient magnitudes instead of minimizing. Defaults to False.
        create_graph (bool, optional): whether to create graph. Defaults to False.
        modify_loss (bool, optional): whether to modify the loss value to make line searches minimize new objective. Defaults to True.
    """
    def __init__(
        self,
        loss_term: float | None = 0,
        relative: Literal['loss_to_grad', 'grad_to_loss'] | None = None,
        graft: Literal['loss_to_grad', 'grad_to_loss'] | None = None,
        square=False,
        mean=True,
        maximize_grad=False,
        create_graph=False,
        modify_loss: bool = True,
    ):
        if (relative is not None) and (graft is not None): warnings.warn('both relative and graft loss are True, they will clash with each other')
        defaults = dict(loss_term=loss_term, relative=relative, graft=graft, square=square, mean=mean, maximize_grad=maximize_grad, create_graph=create_graph, modify_loss=modify_loss)
        super().__init__(defaults)

    @torch.no_grad
    def closure(self, backward, closure, params, var):
        settings = self.settings[params[0]]
        loss_term = settings['loss_term']
        relative = settings['relative']
        graft = settings['graft']
        square = settings['square']
        maximize_grad = settings['maximize_grad']
        create_graph = settings['create_graph']
        modify_loss = settings['modify_loss']
        mean = settings['mean']

        with torch.enable_grad():
            for p in params: p.grad = None
            loss = closure(False)
            grads = TensorList(torch.autograd.grad(loss, params, create_graph=True))

            if square: grads = grads ** 2
            else: grads = grads.abs()

            if mean: f = grads.global_mean()
            else: f = grads.global_sum()


            if graft == 'grad_to_loss': f = f * (loss.detach()/f.detach()).detach()
            if relative == 'grad_to_loss': f = f * loss

            if loss_term is not None and loss_term != 0:
                if relative == 'loss_to_grad': loss_term = loss_term * f
                l = loss
                if graft == 'loss_to_grad': l = loss * (f.detach()/loss.detach()).detach()
                f = f + l*loss_term

            if maximize_grad: f = -f
            if modify_loss: loss = f

            grad = None
            if backward:
                for p in params: p.grad = None
                grad = TensorList(torch.autograd.grad(f, params, create_graph=create_graph))

        return loss, grad