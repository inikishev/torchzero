import math
from collections import deque
from collections.abc import Callable
from typing import Literal

import torch

from ...core import Chainable, Module, apply_transform
from ...utils import Distributions, TensorList, vec_to_tensors
from ...utils.linalg.linear_operator import Sketched
from .newton import _newton_step

def _qr_orthonormalize(A:torch.Tensor):
    m,n = A.shape
    if m < n:
        q, _ = torch.linalg.qr(A.T) # pylint:disable=not-callable
        return q.T
    else:
        q, _ = torch.linalg.qr(A) # pylint:disable=not-callable
        return q

def _orthonormal_sketch(m, n, dtype, device, generator):
    return _qr_orthonormalize(torch.randn(m, n, dtype=dtype, device=device, generator=generator))

def _gaussian_sketch(m, n, dtype, device, generator):
    return torch.randn(m, n, dtype=dtype, device=device, generator=generator) / math.sqrt(m)

class RSN(Module):
    """Randomized Subspace Newton. Performs a Newton step in a random subspace.

    Args:
        sketch_size (int):
            size of the random sketch. This many hessian-vector products will need to be evaluated each step.
        sketch_type (str, optional):
            - "orthonormal" - random orthonormal basis. Orthonormality is necessary to use linear operator based modules such as trust region, but it can be slower to compute.
            - "gaussian" - random gaussian (not orthonormal) basis.
            - "common_directions" - uses history steepest descent directions as the basis[2]. It is orthonormalized on-line using Gram-Schmidt.
            - "mixed" - random orthonormal basis but with three directions set to gradient, slow EMA and fast EMA (default).
        damping (float, optional): hessian damping (scale of identity matrix added to hessian). Defaults to 0.
        hvp_method (str, optional):
            How to compute hessian-matrix product:
            - "batched" - uses batched autograd
            - "autograd" - uses unbatched autograd
            - "forward" - uses finite difference with forward formula, performing 1 backward pass per Hvp.
            - "central" - uses finite difference with a more accurate central formula, performing 2 backward passes per Hvp.

            . Defaults to "batched".
        h (float, optional): finite difference step size. Defaults to 1e-2.
        use_lstsq (bool, optional): whether to use least squares to solve ``Hx=g``. Defaults to False.
        update_freq (int, optional): frequency of updating the hessian. Defaults to 1.
        H_tfm (Callable | None, optional):
            optional hessian transforms, takes in two arguments - `(hessian, gradient)`.

            must return either a tuple: `(hessian, is_inverted)` with transformed hessian and a boolean value
            which must be True if transform inverted the hessian and False otherwise.

            Or it returns a single tensor which is used as the update.

            Defaults to None.
        eigval_fn (Callable | None, optional):
            optional eigenvalues transform, for example ``torch.abs`` or ``lambda L: torch.clip(L, min=1e-8)``.
            If this is specified, eigendecomposition will be used to invert the hessian.
        seed (int | None, optional): seed for random generator. Defaults to None.
        inner (Chainable | None, optional): preconditions output of this module. Defaults to None.

    ### Examples

    RSN with line search
    ```python
    opt = tz.Modular(
        model.parameters(),
        tz.m.RSN(),
        tz.m.Backtracking()
    )
    ```

    RSN with trust region
    ```python
    opt = tz.Modular(
        model.parameters(),
        tz.m.LevenbergMarquardt(tz.m.RSN()),
    )
    ```


    References:
        1. [Gower, Robert, et al. "RSN: randomized subspace Newton." Advances in Neural Information Processing Systems 32 (2019).](https://arxiv.org/abs/1905.10874)
        2. Wang, Po-Wei, Ching-pei Lee, and Chih-Jen Lin. "The common-directions method for regularized empirical risk minimization." Journal of Machine Learning Research 20.58 (2019): 1-49.
    """

    def __init__(
        self,
        sketch_size: int,
        sketch_type: Literal["orthonormal", "gaussian", "common_directions", "mixed"] = "mixed",
        damping:float=0,
        hvp_method: Literal["batched", "autograd", "forward", "central"] = "batched",
        h: float = 1e-2,
        use_lstsq: bool = True,
        update_freq: int = 1,
        H_tfm: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, bool]] | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        eigval_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed: int | None = None,
        inner: Chainable | None = None,
    ):
        defaults = dict(sketch_size=sketch_size, sketch_type=sketch_type,seed=seed,hvp_method=hvp_method, h=h, damping=damping, use_lstsq=use_lstsq, H_tfm=H_tfm, eigval_fn=eigval_fn, update_freq=update_freq)
        super().__init__(defaults)

        if inner is not None:
            self.set_child("inner", inner)

    @torch.no_grad
    def update(self, var):
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        if step % self.defaults['update_freq'] == 0:

            closure = var.closure
            if closure is None:
                raise RuntimeError("RSN requires closure")
            params = var.params
            generator = self.get_generator(params[0].device, self.defaults["seed"])

            ndim = sum(p.numel() for p in params)

            device=params[0].device
            dtype=params[0].dtype

            # sample sketch matrix S: (ndim, sketch_size)
            sketch_size = min(self.defaults["sketch_size"], ndim)
            sketch_type = self.defaults["sketch_type"]
            hvp_method = self.defaults["hvp_method"]

            if sketch_type in ('normal', 'gaussian'):
                S = _gaussian_sketch(ndim, sketch_size, device=device, dtype=dtype, generator=generator)

            elif sketch_type == 'orthonormal':
                S = _orthonormal_sketch(ndim, sketch_size, device=device, dtype=dtype, generator=generator)

            elif sketch_type == 'common_directions':
                # Wang, Po-Wei, Ching-pei Lee, and Chih-Jen Lin. "The common-directions method for regularized empirical risk minimization." Journal of Machine Learning Research 20.58 (2019): 1-49.
                g_list = var.get_grad(create_graph=hvp_method in ("batched", "autograd"))
                g = torch.cat([t.ravel() for t in g_list])

                # initialize directions deque
                if "directions" not in self.global_state:

                    g_norm = torch.linalg.vector_norm(g) # pylint:disable=not-callable
                    if g_norm < torch.finfo(g.dtype).tiny * 2:
                        g = torch.randn_like(g)
                        g_norm = torch.linalg.vector_norm(g) # pylint:disable=not-callable

                    self.global_state["directions"] = deque([g / g_norm], maxlen=sketch_size)
                    S = self.global_state["directions"][0].unsqueeze(1)

                # add new steepest descent direction orthonormal to existing columns
                else:
                    S = torch.stack(tuple(self.global_state["directions"]), dim=1)
                    p = g - S @ (S.T @ g)
                    p_norm = torch.linalg.vector_norm(p) # pylint:disable=not-callable
                    if p_norm > torch.finfo(p.dtype).tiny * 2:
                        p = p / p_norm
                        self.global_state["directions"].append(p)
                        S = torch.cat([S, p.unsqueeze(1)], dim=1)

            elif sketch_type == "mixed":
                g_list = var.get_grad(create_graph=hvp_method in ("batched", "autograd"))
                g = torch.cat([t.ravel() for t in g_list])

                if "slow_ema" not in self.global_state:
                    self.global_state["slow_ema"] = torch.randn_like(g) * 1e-2
                    self.global_state["fast_ema"] = torch.randn_like(g) * 1e-2

                slow_ema = self.global_state["slow_ema"]
                fast_ema = self.global_state["fast_ema"]
                slow_ema.lerp_(g, 0.001)
                fast_ema.lerp_(g, 0.1)

                S = torch.stack([g, slow_ema, fast_ema], dim=1)
                if sketch_size > 3:
                    S_random = _gaussian_sketch(ndim, sketch_size - 3, device=device, dtype=dtype, generator=generator)
                    S = torch.cat([S, S_random], dim=1)

                S = _qr_orthonormalize(S)

            else:
                raise ValueError(f'Unknown sketch_type {sketch_type}')

            # form sketched hessian
            HS, _ = var.hessian_matrix_product(S, at_x0=True, rgrad=None, hvp_method=self.defaults["hvp_method"], normalize=True, retain_graph=False, h=self.defaults["h"])
            H_sketched = S.T @ HS

            self.global_state["H_sketched"] = H_sketched
            self.global_state["S"] = S

    def apply(self, var):
        S: torch.Tensor = self.global_state["S"]
        d_proj = _newton_step(
            var=var,
            H=self.global_state["H_sketched"],
            damping=self.defaults["damping"],
            inner=self.children.get("inner", None),
            H_tfm=self.defaults["H_tfm"],
            eigval_fn=self.defaults["eigval_fn"],
            use_lstsq=self.defaults["use_lstsq"],
            g_proj = lambda g: S.T @ g
        )
        d = S @ d_proj
        var.update = vec_to_tensors(d, var.params)

        return var

    def get_H(self, var=...):
        eigval_fn = self.defaults["eigval_fn"]
        H_sketched: torch.Tensor = self.global_state["H_sketched"]
        S: torch.Tensor = self.global_state["S"]

        if eigval_fn is not None:
            try:
                L, Q = torch.linalg.eigh(H_sketched) # pylint:disable=not-callable
                L: torch.Tensor = eigval_fn(L)
                H_sketched = Q @ L.diag_embed() @ Q.mH

            except torch.linalg.LinAlgError:
                pass

        return Sketched(S, H_sketched)
