from typing import Literal
from collections.abc import Callable
import torch

from ...core import Module, Target, Transform, Chainable, apply_transform
from ...utils import NumberList, TensorList, as_tensorlist
from ...utils.derivatives import hvp, hvp_fd_forward, hvp_fd_central

def sophia_H(
    tensors: TensorList,
    h: TensorList | None,
    exp_avg_: TensorList,
    h_exp_avg_: TensorList,
    beta1: float | NumberList,
    beta2: float | NumberList,
    update_freq: int,
    precond_scale: float | NumberList,
    clip: float | NumberList,
    eps: float | NumberList,
    step: int
):
    # momentum
    exp_avg_.lerp_(tensors, 1-beta1)

    # update preconditioner
    if step % update_freq == 0:
        assert h is not None
        h_exp_avg_.lerp_(h, 1-beta2)

    else:
        assert h is None

    denom = (h_exp_avg_ * precond_scale).clip_(min=eps)
    return (exp_avg_ / denom).clip_(-clip, clip)


class SophiaH(Module):
    """SophiaH optimizer from https://arxiv.org/abs/2305.14342

    Args:
        beta1 (float, optional): first momentum. Defaults to 0.96.
        beta2 (float, optional): momentum for hessian diagonal estimate. Defaults to 0.99.
        update_freq (int, optional):
            frequency of updating hessian diagonal estimate via a hessian-vector product. Defaults to 10.
        precond_scale (float, optional):
            scale of the preconditioner. Defaults to 1.
        clip (float, optional):
            clips update to (-clip, clip). Defaults to 1.
        eps (float, optional):
            clips hessian diagonal esimate to be no less than this value. Defaults to 1e-12.
        hvp_method (str, optional):
            determines how hessian-vector products are evaluated.

            - "autograd" - use pytorch autograd to calculate hessian-vector products.
            - "forward" - use two gradient evaluations to estimate hessian-vector products via froward finite differnce formula.
            - "central" - uses three gradient evaluations to estimate hessian-vector products via central finite differnce formula.
            Defaults to "autograd".
        h (float, optional): finite difference step size if :code:`hvp_method` is "forward" or "central". Defaults to 1e-3.
        n_samples (int, optional):
            number of hessian-vector products with random vectors to evaluate each time when updating
            the preconditioner. Larger values may lead to better hessian diagonal estimate. Defaults to 1.
        seed (int | None, optional): seed for random vectors. Defaults to None.
        inner (Chainable | None, optional): preconditioning is applied to the output of this module. Defaults to None.
    """
    def __init__(
        self,
        beta1: float = 0.96,
        beta2: float = 0.99,
        update_freq: int = 10,
        precond_scale: float = 1,
        clip: float = 1,
        eps: float = 1e-12,
        hvp_method: Literal['autograd', 'forward', 'central'] = 'autograd',
        fd_h: float = 1e-3,
        n_samples = 1,
        seed: int | None = None,
        inner: Chainable | None = None
    ):
        defaults = dict(beta1=beta1, beta2=beta2, update_freq=update_freq, precond_scale=precond_scale, clip=clip, eps=eps, hvp_method=hvp_method, n_samples=n_samples, fd_h=fd_h, seed=seed)
        super().__init__(defaults)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def step(self, var):
        params = var.params
        settings = self.settings[params[0]]
        hvp_method = settings['hvp_method']
        fd_h = settings['fd_h']
        update_freq = settings['update_freq']
        n_samples = settings['n_samples']

        seed = settings['seed']
        generator = None
        if seed is not None:
            if 'generator' not in self.global_state:
                self.global_state['generator'] = torch.Generator(params[0].device).manual_seed(seed)
            generator = self.global_state['generator']

        beta1, beta2, precond_scale, clip, eps = self.get_settings(params,
            'beta1', 'beta2', 'precond_scale', 'clip', 'eps', cls=NumberList)

        exp_avg, h_exp_avg = self.get_state(params, 'exp_avg', 'h_exp_avg', cls=TensorList)

        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        closure = var.closure
        assert closure is not None

        h = None
        if step % update_freq == 0:

            grad=None
            for i in range(n_samples):
                u = [torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=generator) for p in params]

                if hvp_method == 'autograd':
                    if grad is None: grad = var.get_grad(create_graph=True)
                    assert grad is not None
                    Hvp = hvp(params, grad, u, retain_graph=i < n_samples-1)

                elif hvp_method == 'forward':
                    loss, Hvp = hvp_fd_forward(closure, params, u, h=fd_h, g_0=var.get_grad(), normalize=True)

                elif hvp_method == 'central':
                    loss, Hvp = hvp_fd_central(closure, params, u, h=fd_h, normalize=True)

                else:
                    raise ValueError(hvp_method)

                if h is None: h = Hvp
                else: torch._foreach_add_(h, Hvp)

            assert h is not None
            if n_samples > 1: torch._foreach_div_(h, n_samples)

        update = var.get_update()
        if 'inner' in self.children:
            update = apply_transform(self.children['inner'], tensors=update, params=params, grads=var.grad, var=var)

        var.update = sophia_H(
            tensors=TensorList(update),
            h=TensorList(h) if h is not None else None,
            exp_avg_=exp_avg,
            h_exp_avg_=h_exp_avg,
            beta1=beta1,
            beta2=beta2,
            update_freq=update_freq,
            precond_scale=precond_scale,
            clip=clip,
            eps=eps,
            step=step,
        )
        return var
