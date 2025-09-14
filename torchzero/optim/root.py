"""WIP, untested"""
from collections.abc import Callable

from abc import abstractmethod
import torch
from ..modules.second_order.multipoint import sixth_order_3p, sixth_order_5p, two_point_newton, sixth_order_3pm2, _solve

def make_evaluate(f: Callable[[torch.Tensor], torch.Tensor]):
    def evaluate(x, order) -> tuple[torch.Tensor, ...]:
        """order=0 - returns (f,), order=1 - returns (f, J), order=2 - returns (f, J, H), etc."""
        n = x.numel()

        if order == 0:
            f_x = f(x)
            return (f_x, )

        x.requires_grad_()
        with torch.enable_grad():
            f_x = f(x)
            I = torch.eye(n, device=x.device, dtype=x.dtype),
            g_x = torch.autograd.grad(f_x, x, I, create_graph=order!=1, is_grads_batched=True)[0]
            ret = [f_x, g_x]
            T = g_x

            # get all derivative up to order
            for o in range(2, order + 1):
                is_last = o == order
                I = torch.eye(T.numel(), device=x.device, dtype=x.dtype),
                T = torch.autograd.grad(T.ravel(), x, I, create_graph=not is_last, is_grads_batched=True)[0]
                ret.append(T.view(n, n, *T.shape[1:]))

        return tuple(ret)

    return evaluate

class RootBase:
    @abstractmethod
    def one_iteration(
        self,
        x: torch.Tensor,
        evaluate: Callable[[torch.Tensor, int], tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        """"""


# ---------------------------------- methods --------------------------------- #
def newton(x:torch.Tensor, f_j, lstsq:bool=False):
    f_x, G_x = f_j(x)
    return x - _solve(G_x, f_x, lstsq=lstsq)

class Newton(RootBase):
    def __init__(self, lstsq: bool=False): self.lstsq = lstsq
    def one_iteration(self, x, evaluate): return newton(x, evaluate, self.lstsq)


class SixthOrder3P(RootBase):
    """sixth-order iterative method

    Abro, Hameer Akhtar, and Muhammad Mujtaba Shaikh. "A new time-efficient and convergent nonlinear solver." Applied Mathematics and Computation 355 (2019): 516-536.
    """
    def __init__(self, lstsq: bool=False): self.lstsq = lstsq
    def one_iteration(self, x, evaluate):
        def f(x): return evaluate(x, 0)[0]
        def f_j(x): return evaluate(x, 1)
        return sixth_order_3p(x, f, f_j, self.lstsq)
