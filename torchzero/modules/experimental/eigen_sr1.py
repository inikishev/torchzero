import torch

from ...core import Transform
from ...linalg.orthogonalize import orthogonalize, OrthogonalizeMethod
from ...linalg.eigh import eigh_plus_uuT, regularize_eigh
from ...utils import TensorList, unpack_states, vec_to_tensors_
from ..opt_utils import safe_clip
from .eigengrad import _eigengrad_update_state_, eigengrad_apply


def sr1_u(L: torch.Tensor, Q: torch.Tensor, s:torch.Tensor, y: torch.Tensor, tol:float):
    """u from u u^T correction and its sign"""
    r = y - torch.linalg.multi_dot([Q, L.diag_embed(), Q.T, s]) # pylint:disable=not-callable
    rs = r.dot(s)

    if rs.abs() < tol * torch.linalg.vector_norm(r) * torch.linalg.vector_norm(s): # pylint:disable=not-callable
        return None, None

    u = r / rs.abs().sqrt()
    return u, torch.sign(rs)

class EigenSR1(Transform):
    def __init__(
        self,
        rank: int = 100,
        tol: float = 1e-32,
        eig_tol: float | None = None,
        damping: float = 0,
        rdamping: float = 0,
        abs: bool = False,
        mm_tol: float = 1e-7,
        mm_truncate: int | None = None,
        mm_damping: float = 1e-4,
        mm_rdamping: float = 0,
        mm_abs: bool = True,
        id_reg: float | None = None,
        column_space_tol=1e-9,
        beta: float = 0.95,
        balance_tol: float = 10,
        balance_strength: float = 1e-1,

        eigenbasis_optimizer = None,
        update_freq: int = 1,
        init_steps: int = 10,
        orthogonalize_interval: int | None = 1,
        orthogonalize_method: OrthogonalizeMethod = 'qr',

        hvp_method = "autograd",
        h = 1e-3,
        inner = None,

    ):
        defaults = locals().copy()
        for k in ["self", "inner"]:
            del defaults[k]

        super().__init__(defaults)

    def update_states(self, objective, states, settings):
        fs = settings[0]
        step = self.increment_counter("step", 0)

        if step % fs["update_freq"] == 0:

            params = TensorList(objective.params)

            # compute y as hessian-vector product with s (random vecs during init steps)
            if ("p_prev" not in self.global_state) or (step < fs["init_steps"]):
                s_list = params.sample_like('rademacher')

            else:
                p_prev = self.global_state["p_prev"]
                s_list = params - p_prev

            if s_list.dot(s_list) < torch.finfo(s_list[0].dtype).tiny * 2:
                s_list = params.sample_like('rademacher')

            self.global_state["p_prev"] = params

            # compute y as hessian-vector product with s
            Hz_list, _ = objective.hessian_vector_product(s_list, rgrad=None, at_x0=True, hvp_method=fs["hvp_method"], h=fs["h"])

            s = torch.cat([t.ravel() for t in s_list])
            y = torch.cat([t.ravel() for t in Hz_list])

            # keep track of exponential moving average of hessian diagonal and balance eigenvalues
            if (fs["balance_strength"] != 0) and (step > fs["init_steps"]) and ("L" in self.global_state):

                D = s * y # hutchinson estimator
                exp_avg = self.global_state.get("exp_avg", None)

                if exp_avg is None:
                    exp_avg = self.global_state["exp_avg"] = D

                else:
                    exp_avg.lerp_(D, weight=1-fs["beta"])

                L = self.global_state["L"]
                L_abs = L.abs()
                tau = L_abs.amax() / exp_avg.abs().amax()

                if tau > fs["balance_tol"]:
                    L_balanced = L_abs.pow((1 / tau) ** (1 / fs["balance_strength"])).copysign(L)
                    self.global_state["L"] = torch.where(L_abs > 1, L_balanced, L)

            # initialize L and Q on 1st step
            if "L" not in self.global_state:

                L = torch.zeros(1, dtype=s.dtype, device=s.device) # rank, rank
                Q = torch.zeros([s.numel(), 1], dtype=s.dtype, device=s.device) # ndim, rank

                u, sign = sr1_u(L=L, Q=Q, s=s, y=y, tol=0)
                assert u is not None and sign is not None

                # for uu^T u is eigenvector and u^T u is eigenvalue
                norm = torch.linalg.vector_norm(u).clip(min=torch.finfo(u.dtype).tiny * 2) # pylint:disable=not-callable

                self.global_state["L"] = self.global_state["L_reg"] = (u.dot(u).unsqueeze(0) / norm) * sign # (rank,)
                self.global_state["Q"] = self.global_state["Q_reg"] = u.unsqueeze(-1) / norm # (m, rank)

            # update hessian
            else:
                try:
                    L = self.global_state["L"]
                    Q = self.global_state["Q"]

                    H_step = self.increment_counter("H_step", start=0)
                    if H_step % fs["orthogonalize_interval"] == 0:
                        Q = orthogonalize(Q, method=fs["orthogonalize_method"])

                    u, sign = sr1_u(L=L, Q=Q, s=s, y=y, tol=fs["tol"])

                    if (u is not None) and (sign is not None):

                        # compute new factors
                        L_new, Q_new = eigh_plus_uuT(L, Q, u, tol=fs["column_space_tol"], alpha=sign.item(), retry_float64=True)

                        # truncate/regularize new factors (those go into the accumulator)
                        L_new, Q_new = regularize_eigh(L=L_new, Q=Q_new, truncate=min(fs["rank"], s.numel()),
                                                      tol=fs["eig_tol"], damping=fs["damping"], rdamping=fs["rdamping"])

                        _eigengrad_update_state_(state=self.global_state, setting=fs, L_new=L_new, Q_new=Q_new)

                except torch.linalg.LinAlgError:
                    pass



    def apply_states(self, objective, states, settings):
        fs = settings[0]
        updates = objective.get_updates()

        if "eigenbasis_state" not in self.global_state:
            self.global_state["eigenbasis_state"] = {}

        step = self.global_state["step"] # starts at 0
        if step < fs["init_steps"]:

            # skip update first init_steps to let hessian kick-start
            objective.stop = True
            objective.skip_update = True
            return objective

        if "L_reg" not in self.global_state:
            TensorList(updates).clip_(-0.1, 0.1)
            return objective

        dir = eigengrad_apply(
            tensor = torch.cat([t.ravel() for t in updates]),
            L_reg = self.global_state["L_reg"],
            Q_reg = self.global_state["Q_reg"],
            beta = None,
            step = None,
            debias = False,
            id_reg = fs["id_reg"],
            eigenbasis_optimizer = fs["eigenbasis_optimizer"],
            eigenbasis_state = self.global_state["eigenbasis_state"],
            whiten_fn = lambda x: x
        )

        vec_to_tensors_(dir, updates)
        return objective