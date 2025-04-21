import torch

from ...core import Transform
from ...utils import TensorList, as_tensorlist

# ------------------------------- Polak-Ribière ------------------------------ #
class PolakRibiere(Transform):
    """Polak-Ribière-Polyak nonlinear conjugate gradient method. This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe` after this."""
    def __init__(self):
        super().__init__(defaults={}, uses_grad=False)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        grad = as_tensorlist(target) # for brevity

        prev_dir, prev_grad = self.get_state('prev_dir', 'prev_grad', params=params, cls=TensorList, init=[torch.zeros_like, grad])

        denom = prev_grad.dot(prev_grad)
        if denom == 0: beta = 0
        else: beta = (grad.dot(grad - prev_grad) / denom).clamp(min=0)

        dir = grad.add_(prev_dir.mul_(beta))
        prev_dir.copy_(dir)
        prev_grad.set_(grad)
        return dir

# ------------------------------ Fletcher–Reeves ----------------------------- #
class FletcherReeves(Transform):
    """Fletcher–Reeves nonlinear conjugate gradient method. This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe` after this."""
    def __init__(self):
        super().__init__(defaults={}, uses_grad=False)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        grad = as_tensorlist(target) # for brevity

        if 'prev_dot' not in self.global_state:
            self.global_state['prev_dot'] = grad.dot(grad)

        prev_dir = self.get_state('prev_dir', params=params, cls=TensorList)
        prev_dot = self.global_state['prev_dot']

        cur_dot = grad.dot(grad)
        if prev_dot == 0: beta = 0
        else: beta = cur_dot / prev_dot

        dir = grad.add_(prev_dir.mul_(beta))
        prev_dir.copy_(dir)
        self.global_state['prev_dot'] = cur_dot
        return dir


# ----------------------------- Hestenes–Stiefel ----------------------------- #
def hestenes_stiefel_beta(grad: TensorList, prev_dir: TensorList,prev_grad: TensorList):
    grad_diff = grad - prev_grad
    denom = prev_dir.dot(grad_diff)
    if denom == 0: return 0
    return (grad.dot(grad_diff) / denom).neg()

def hestenes_stiefel_(grad: TensorList, prev_dir_: TensorList,prev_grad_: TensorList):
    beta = hestenes_stiefel_beta(grad, prev_dir=prev_dir_, prev_grad=prev_grad_)

    dir = grad.add_(prev_dir_.mul_(beta))
    prev_dir_.copy_(dir)
    prev_grad_.set_(grad)
    return dir

class HestenesStiefel(Transform):
    """Hestenes–Stiefel nonlinear conjugate gradient method. This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe` after this."""
    def __init__(self):
        super().__init__(defaults={}, uses_grad=False)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        prev_dir, prev_grad = self.get_state('prev_dir', 'prev_grad', params=params, cls=TensorList, init=[torch.zeros_like, target])
        return hestenes_stiefel_(as_tensorlist(target), prev_dir_=prev_dir, prev_grad_=prev_grad)


# --------------------------------- Dai–Yuan --------------------------------- #
def dai_yuan_beta(grad: TensorList, prev_dir: TensorList,prev_grad: TensorList):
    denom = prev_dir.dot(grad - prev_grad)
    if denom == 0: return 0
    return (grad.dot(grad) / denom).neg()

def dai_yuan_(grad: TensorList, prev_dir_: TensorList,prev_grad_: TensorList):
    beta = dai_yuan_beta(grad, prev_dir=prev_dir_, prev_grad=prev_grad_)

    dir = grad.add_(prev_dir_.mul_(beta))
    prev_dir_.copy_(dir)
    prev_grad_.set_(grad)
    return dir

class DaiYuan(Transform):
    """Dai–Yuan nonlinear conjugate gradient method. This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe` after this."""
    def __init__(self):
        super().__init__(defaults={}, uses_grad=False)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        prev_dir, prev_grad = self.get_state('prev_dir', 'prev_grad', params=params, cls=TensorList, init=[torch.zeros_like, target])
        return dai_yuan_(as_tensorlist(target), prev_dir_=prev_dir, prev_grad_=prev_grad)


# -------------------------------- Liu-Storey -------------------------------- #
class LiuStorey(Transform):
    """Liu-Storey nonlinear conjugate gradient method. This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe` after this."""
    def __init__(self):
        super().__init__(defaults={}, uses_grad=False)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        grad = as_tensorlist(target) # for brevity

        prev_dir, prev_grad = self.get_state('prev_dir', 'prev_grad', params=params, cls=TensorList, init=[torch.zeros_like, grad])

        denom = prev_grad.dot(prev_dir)
        if denom == 0: beta = 0
        else: beta = grad.dot(grad - prev_grad) / denom

        dir = grad.add_(prev_dir.mul_(beta))
        prev_dir.copy_(dir)
        prev_grad.set_(grad)
        return dir

# ----------------------------- Conjugate Descent ---------------------------- #
class ConjugateDescent(Transform):
    """Conjugate Descent (CD). This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe` after this."""
    def __init__(self):
        super().__init__(defaults={}, uses_grad=False)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        grad = as_tensorlist(target) # for brevity

        prev_dir = self.get_state('prev_dir', params=params, cls=TensorList, init = [torch.zeros_like, grad])
        if 'denom' not in self.global_state:
            self.global_state['denom'] = torch.tensor(0.).to(grad[0])

        denom = self.global_state['denom'] # prev_grad.T @ prev_dir
        if denom == 0: beta = 0
        else: beta = grad.dot(grad) / denom

        dir = grad.add_(prev_dir.mul_(beta))
        prev_dir.copy_(dir)
        self.global_state['denom'] = grad.dot(dir)
        return dir


# -------------------------------- Hager-Zhang ------------------------------- #
class HagerZhang(Transform):
    """Hager-Zhang nonlinear conjugate gradient method,
    it's the most complicated one so surely it must be better that the other ones.
    This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe` after this."""
    def __init__(self):
        super().__init__(defaults={}, uses_grad=False)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        grad = as_tensorlist(target) # for brevity

        prev_dir, prev_grad = self.get_state('prev_dir', 'prev_grad', params=params, cls=TensorList, init=[torch.zeros_like, grad])
        grad_diff = grad - prev_grad

        # term 1
        denom = prev_dir.dot(grad_diff)
        if denom == 0: beta = 0
        else:
            term1 = 1/denom

            # term2
            term2 = (grad_diff - (2 * prev_dir * (grad_diff.pow(2).global_sum()/denom))).dot(grad)

            beta = (term1 * term2).neg()

        dir = grad.add_(prev_dir.mul_(beta))
        prev_dir.copy_(dir)
        prev_grad.set_(grad)
        return dir

# ----------------------------------- HS-DY ---------------------------------- #
def hs_dy_beta(grad: TensorList, prev_dir: TensorList,prev_grad: TensorList):
    grad_diff = grad - prev_grad
    denom = prev_dir.dot(grad_diff)
    if denom == 0: return 0

    # Dai-Yuan
    dy_beta = (grad.dot(grad) / denom).neg().clamp(min=0)

    # Hestenes–Stiefel
    hs_beta = (grad.dot(grad_diff) / denom).neg().clamp(min=0)

    return max(0, min(dy_beta, hs_beta)) # type:ignore

def hd_dy_(grad: TensorList, prev_dir_: TensorList,prev_grad_: TensorList):
    beta = hs_dy_beta(grad, prev_dir=prev_dir_, prev_grad=prev_grad_)

    dir = grad.add_(prev_dir_.mul_(beta))
    prev_dir_.copy_(dir)
    prev_grad_.set_(grad)
    return dir

class HybridHS_DY(Transform):
    """HS-DY hybrid conjugate gradient method.
    This requires step size to be determined via a line search, so put a line search like :code:`StrongWolfe` after this."""
    def __init__(self):
        super().__init__(defaults={}, uses_grad=False)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        prev_dir, prev_grad = self.get_state('prev_dir', 'prev_grad', params=params, cls=TensorList, init=[torch.zeros_like, target])
        return hd_dy_(as_tensorlist(target), prev_dir_=prev_dir, prev_grad_=prev_grad)
