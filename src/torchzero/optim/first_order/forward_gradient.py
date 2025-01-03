from typing import Literal
import torch
from ..modular import Modular
from ...core import OptimizationState, OptimizerModule
from ...tensorlist import Distributions

class ForwardGradientsModular(Modular):
    """EXPERIMENTAL (WILL TEST TOMORROW).

    Evaluates jacobian-vector product with a random vector using forward mode autodiff (torch.func.jvp), which is
    the true directional derivative in the direction of that vector.

    This requires the closure to be rewritten with functional_call:

    .. code:: python
        def closure(params):
            preds = torch.func.functional_call(model, params, (inputs, ))
            loss = loss_fn(preds, targets)
            return loss


    This is a subclass of Modular (temporarily) so you have to pass modules to it.
    For example:
    .. code:: python
        import torchzero as tz
        opt = ForwardGradientsModular(model, tz.m.LR(1e-2))

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        modules: list of OptimizerModules.
        distribution (Distributions, optional): distribution for directional derivative vector. Defaults to 'normal'.

    Reference:
        Baydin, A. G., Pearlmutter, B. A., Syme, D., Wood, F., & Torr, P. (2022).
        Gradients without backpropagation. arXiv preprint arXiv:2202.08587.
        https://arxiv.org/abs/2202.08587
    """
    def __init__(self, model:torch.nn.Module, *modules: OptimizerModule, distribution: Distributions = 'normal'):
        if not isinstance(model, torch.nn.Module): raise TypeError("model must be torch.nn.Module")
        super().__init__(model, *modules)
        self.distribuition: Distributions = distribution

    @torch.no_grad
    def step(self, closure): # type:ignore # pylint:disable=signature-differs
        assert self.model is not None
        keys = [k for k, v in self.model.named_parameters() if v.requires_grad]

        def list_closure(list_params):
            dict_params = {k: p for k, p in zip(keys, list_params)}
            return closure(dict_params)

        params = self.get_params()
        vec = params.sample_like(1, distribution = self.distribuition)

        def forward_grad_closure(backward=True):
            if backward:
                loss, directional_derivative = torch.func.jvp(list_closure, primals = tuple(params), tangents = tuple(vec)) # type:ignore
                ascent = vec * directional_derivative
                params.set_grad_(ascent)
            else: loss = list_closure(params)
            return loss

        state = OptimizationState(forward_grad_closure, self.model)
        return self.chain.step(state)


class ForwardGradientsModularv2(Modular):
    """uses torch.func with swap_tensors. Requires no custom closure, however this clones parameters
    on each closure evaluation.
    """
    def __init__(self, params, *modules, distribution: Distributions = 'normal', mode: Literal['jvp', 'dot'] = 'jvp'):
        #if not isinstance(model, torch.nn.Module): raise TypeError("model must be torch.nn.Module")
        super().__init__(params, *modules)
        self.distribuition: Distributions = distribution
        self.mode = mode

    @torch.no_grad
    def step(self, closure): # type:ignore # pylint:disable=signature-differs
        params = self.get_params()

        def param_closure(*new_params):
            # swap params to new params
            for old_p, new_p in zip(params, new_params):
                torch.utils.swap_tensors(old_p, new_p)

            value = closure(False)

            # swap params back to original ones
            for old_p, new_p in zip(params, new_params):
                torch.utils.swap_tensors(new_p, old_p)

            return value

        vec = params.sample_like(1, distribution = self.distribuition)

        def forward_grad_closure(backward=True):
            if backward:
                if self.mode == 'jvp':
                    loss, directional_derivative = torch.func.jvp(param_closure, primals = tuple(params.clone().detach_()), tangents = tuple(vec),) # type:ignore
                    ascent = vec * directional_derivative
                    params.set_grad_(ascent)
                else:
                    loss = closure(True)
                    directional_derivative = params.grad.mul_(vec).sum()
                    params.set_grad_(vec * directional_derivative)
            else: loss = closure(False)
            return loss

        state = OptimizationState(forward_grad_closure, self.model)
        return self.chain.step(state)