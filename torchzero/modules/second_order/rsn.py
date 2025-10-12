import math
from collections import deque
from collections.abc import Callable
from typing import Literal

import torch

from ...core import Chainable, Transform, HVPMethod
from ...utils import vec_to_tensors_
from ...linalg.linear_operator import Sketched

from .newton import _newton_update_state_, _newton_solve

def _qr_orthonormalize(A:torch.Tensor):
    m,n = A.shape
    if m < n:
        q, _ = torch.linalg.qr(A.T) # pylint:disable=not-callable
        return q.T

    q, _ = torch.linalg.qr(A) # pylint:disable=not-callable
    return q


def _orthonormal_sketch(m, n, dtype, device, generator):
    return _qr_orthonormalize(torch.randn(m, n, dtype=dtype, device=device, generator=generator))

def _rademacher_sketch(m, n, dtype, device, generator):
    rademacher = torch.bernoulli(torch.full((m,n), 0.5, device=device, dtype=dtype), generator = generator).mul_(2).sub_(1)
    return rademacher.mul_(1 / math.sqrt(m))

def _row_sketch(m, n, dtype, device, generator):
    weights = torch.ones(m, dtype=dtype, device=device)
    indices = torch.multinomial(weights, n, replacement=False, generator=generator)

    P = torch.zeros(m, n, dtype=dtype, device=device)
    P[indices, range(n)] = 1
    return P

def _topk_rows(grad, m, n, dtype, device, generator):
    _, indices = torch.topk(grad.abs(), n)
    P = torch.zeros(m, n, dtype=dtype, device=device)
    P[indices, range(n)] = 1
    return P

class SubspaceNewton(Transform):
    """Subspace Newton. Performs a Newton step in a subspace (random or spanned by past gradients).

    Args:
        sketch_size (int):
            size of the random sketch. This many hessian-vector products will need to be evaluated each step.
        sketch_type (str, optional):
            - "common_directions" - uses history steepest descent directions as the basis[2]. It is orthonormalized on-line using Gram-Schmidt (default).
            - "orthonormal" - random orthonormal basis. Orthonormality is necessary to use linear operator based modules such as trust region, but it can be slower to compute.
            - "rows" - samples random rows.
            - "topk" - samples top-rank rows with largest gradient magnitude.
            - "rademacher" - approximately orthonormal (if dimension is large) scaled random rademacher basis.
            - "mixed" - random orthonormal basis but with four directions set to gradient, slow and fast gradient EMAs, and previous update direction.
        damping (float, optional): hessian damping (scale of identity matrix added to hessian). Defaults to 0.
        hvp_method (str, optional):
            How to compute hessian-matrix product:
            - "batched_autograd" - uses batched autograd
            - "autograd" - uses unbatched autograd
            - "forward" - uses finite difference with forward formula, performing 1 backward pass per Hvp.
            - "central" - uses finite difference with a more accurate central formula, performing 2 backward passes per Hvp.

            . Defaults to "batched_autograd".
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
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.RSN(),
        tz.m.Backtracking()
    )
    ```

    RSN with trust region
    ```python
    opt = tz.Optimizer(
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
        sketch_size: int = 100,
        sketch_type: Literal["orthonormal", "common_directions", "mixed", "rademacher", "rows", "topk"] = "common_directions",
        damping:float=0,
        eigval_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        eigv_tol: float | None = None,
        truncate: int | None = None,
        update_freq: int = 1,
        precompute_inverse: bool = False,
        use_lstsq: bool = False,
        hvp_method: HVPMethod = "batched_autograd",
        h: float = 1e-2,
        seed: int | None = None,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner'], defaults["update_freq"]
        super().__init__(defaults, update_freq=update_freq, inner=inner)

    @torch.no_grad
    def update_states(self, objective, states, settings):
        fs = settings[0]
        params = objective.params
        generator = self.get_generator(params[0].device, fs["seed"])

        ndim = sum(p.numel() for p in params)

        device=params[0].device
        dtype=params[0].dtype

        # sample sketch matrix S: (ndim, sketch_size)
        sketch_size = min(fs["sketch_size"], ndim)
        sketch_type = fs["sketch_type"]
        hvp_method = fs["hvp_method"]

        if sketch_type == "rademacher":
            S = _rademacher_sketch(ndim, sketch_size, device=device, dtype=dtype, generator=generator)

        elif sketch_type == 'orthonormal':
            S = _orthonormal_sketch(ndim, sketch_size, device=device, dtype=dtype, generator=generator)

        elif sketch_type == "rows":
            S = _row_sketch(ndim, sketch_size, device=device, dtype=dtype, generator=generator)

        elif sketch_type == "topk":
            g_list = objective.get_grads(create_graph=hvp_method in ("batched_autograd", "autograd"))
            g = torch.cat([t.ravel() for t in g_list])
            S = _topk_rows(g, ndim, sketch_size, device=device, dtype=dtype, generator=generator)

        elif sketch_type == 'common_directions':
            # Wang, Po-Wei, Ching-pei Lee, and Chih-Jen Lin. "The common-directions method for regularized empirical risk minimization." Journal of Machine Learning Research 20.58 (2019): 1-49.
            g_list = objective.get_grads(create_graph=hvp_method in ("batched_autograd", "autograd"))
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
            g_list = objective.get_grads(create_graph=hvp_method in ("batched_autograd", "autograd"))
            g = torch.cat([t.ravel() for t in g_list])

            # initialize state
            if "slow_ema" not in self.global_state:
                self.global_state["slow_ema"] = torch.randn_like(g) * 1e-2
                self.global_state["fast_ema"] = torch.randn_like(g) * 1e-2
                self.global_state["p_prev"] = torch.randn_like(g)

            # previous update direction
            p_cur = torch.cat([t.ravel() for t in params])
            prev_dir = p_cur - self.global_state["p_prev"]
            self.global_state["p_prev"] = p_cur

            # EMAs
            slow_ema = self.global_state["slow_ema"]
            fast_ema = self.global_state["fast_ema"]
            slow_ema.lerp_(g, 0.001)
            fast_ema.lerp_(g, 0.1)

            # form and orthogonalize sketching matrix
            S = torch.stack([g, slow_ema, fast_ema, prev_dir], dim=1)
            if sketch_size > 4:
                S_random = torch.randn(ndim, sketch_size - 3, device=device, dtype=dtype, generator=generator) / math.sqrt(ndim)
                S = torch.cat([S, S_random], dim=1)

            S = _qr_orthonormalize(S)

        else:
            raise ValueError(f'Unknown sketch_type {sketch_type}')

        # print(f'{S.shape = }')
        # I = torch.eye(S.size(1), device=S.device, dtype=S.dtype)
        # print(f'{torch.nn.functional.mse_loss(S.T @ S, I) = }')

        # form sketched hessian
        HS, _ = objective.hessian_matrix_product(S, rgrad=None, at_x0=True,
                                                 hvp_method=fs["hvp_method"], h=fs["h"])
        H_sketched = S.T @ HS

        # update state
        _newton_update_state_(
            state = self.global_state,
            H = H_sketched,
            damping = fs["damping"],
            eigval_fn = fs["eigval_fn"],
            eigv_tol = fs["eigv_tol"],
            truncate = fs["truncate"],
            precompute_inverse = fs["precompute_inverse"],
            use_lstsq = fs["use_lstsq"]
        )

        self.global_state["S"] = S

    def apply_states(self, objective, states, settings):
        updates = objective.get_updates()
        fs = settings[0]

        S = self.global_state["S"]
        b = torch.cat([t.ravel() for t in updates])
        b_proj = S.T @ b

        d_proj = _newton_solve(b=b_proj, state=self.global_state, use_lstsq=fs["use_lstsq"])

        d = S @ d_proj
        vec_to_tensors_(d, updates)
        return objective

    def get_H(self, objective=...):
        if "H" in self.global_state:
            H_sketched = self.global_state["H"]

        else:
            L = self.global_state["L"]
            Q = self.global_state["Q"]
            H_sketched = Q @ L.diag_embed() @ Q.mH

        S: torch.Tensor = self.global_state["S"]
        return Sketched(S, H_sketched)
