# pylint: disable = non-ascii-name
from collections.abc import Mapping

import torch

from ...core import Chainable, TensorTransform
from ...linalg.eigh import eigh_plus_uuT, regularize_eigh
from ...linalg.orthogonalize import OrthogonalizeMethod, orthogonalize
from ...linalg.linear_operator import Eigendecomposition
from ..adaptive.lre_optimizers import LREOptimizerBase


def _eigengrad_update_state_(state:dict, setting: Mapping, L_new: torch.Tensor | None, Q_new:torch.Tensor | None):
    """stores L, Q, L_reg, Q_reg and reprojects eigenbasis opt (this is also used on other eigen based modules)"""
    if (L_new is not None) and (Q_new is not None):

        # re-orthogonalize
        orthogonalize_interval = setting["orthogonalize_interval"]
        if orthogonalize_interval is not None:
            Q_step = state.get("Q_step", 0)
            state["Q_step"] = Q_step + 1
            if Q_step % orthogonalize_interval == 0:
                Q_new = orthogonalize(Q_new, method=setting["orthogonalize_method"])

        # take absolute value (for hessian)
        if setting.get("abs", False):
            L_new = L_new.abs()

        # store
        state["L"] = L_new
        state["Q"] = Q_new

        # absolute value for matmul
        if setting.get("mm_abs", False):
            L_new = L_new.abs()

        # regularize for matmul
        # this second round of regularization is only used for preconditioning
        # and doesn't affect the accumulator
        L_reg_new, Q_reg_new = regularize_eigh(L=L_new, Q=Q_new,
            truncate=setting["mm_truncate"],
            tol=setting["mm_tol"],
            damping=setting["mm_damping"],
            rdamping=setting["mm_rdamping"],
        )

        # print(f'{state["L_reg"] = }, {L_reg_new = }')

        # reproject eigenbasis optimizer
        if (L_reg_new is not None) and (Q_reg_new is not None):
            eigenbasis_optimizer: LREOptimizerBase | None = setting["eigenbasis_optimizer"]
            if eigenbasis_optimizer is not None:
                eigenbasis_optimizer.reproject(L_old=state["L_reg"], Q_old=state["Q_reg"], L_new=L_reg_new,
                                                Q_new=Q_reg_new, state=state["eigenbasis_state"])

            state["L_reg"] = L_reg_new
            state["Q_reg"] = Q_reg_new


def eigengrad_apply(
    tensor: torch.Tensor,
    L_reg: torch.Tensor,
    Q_reg: torch.Tensor,
    beta: float | None,
    step: int | None,
    debias: bool,
    id_reg: float | None,
    eigenbasis_optimizer: LREOptimizerBase | None,
    eigenbasis_state: dict,

    whiten_fn = torch.sqrt
):
    # debias
    if debias:
        assert beta is not None and step is not None
        L_reg = L_reg / (1 - beta **step)

    # step with eigenbasis optimizer
    if eigenbasis_optimizer is not None:
        if (id_reg is not None) and (id_reg != 0):
            raise RuntimeError("id_reg is not compatible with eigenbasis_optimizer")

        update = eigenbasis_optimizer.step(tensor.ravel(), L=L_reg, Q=Q_reg, state=eigenbasis_state)
        return update.view_as(tensor)

    # or just whiten
    # L_reg = L_reg.clip(min=torch.finfo(L_reg.dtype).tiny * 2)

    if id_reg is None or id_reg == 0:
        G = Eigendecomposition(whiten_fn(L_reg), Q_reg, use_nystrom=False)
        dir = G.solve(tensor.ravel())

    else:
        G = Eigendecomposition(whiten_fn(L_reg), Q_reg, use_nystrom=True)
        dir = G.solve_plus_diag(tensor.ravel(), diag=id_reg)

    return dir.view_as(tensor)

class Eigengrad(TensorTransform):
    """we can easily compute rank 1 symmetric update to a low rank eigendecomposition.
    So this stores covariance matrix as it.


    Args:
        rank (int): maximum allowed rank
        beta (float, optional): beta for covariance matrix exponential moving average. Defaults to 0.95.
        eig_tol (float, optional):
            removes eigenvalues this much smaller than largest eigenvalue when updating the preconditioner. Defaults to 1e-7.
        damping (float, optional):
            added to eigenvalues when updating the preconditioner. Defaults to 1e-8.
        rdamping (float, optional):
            added to eigenvalues when updating the preconditioner, relative to largest eigenvalue. Defaults to 0.
        mm_tol (float, optional):
            removes eigenvalues this much smaller than largest eigenvalue when computing the update. Defaults to 1e-7.
        mm_truncate (int | None, optional):
            uses top k eigenvalues to compute the update. Defaults to None.
        mm_damping (float, optional):
            added to eigenvalues when computing the update. Defaults to 1e-4.
        mm_rdamping (float, optional):
            added to eigenvalues when computing the update, relative to largest eigenvalue. Defaults to 0.
        id_reg (float, optional):
            multiplier to identity matrix added to preconditioner before computing update
            If this value is given, solution from Nyström sketch-and-solve will be used to compute the update.
            This value can't be too small (i.e. less than 1e-5) or the solver will be very unstable. Defaults to None.
        column_space_tol (float, optional):
            tolerance for deciding if new eigenvector is within column space of the covariance matrix. Defaults to 1e-9.
        concat_params (bool, optional):
            whether to precondition all parameters at once if True, or each separately if False. Defaults to True.
        update_freq (int, optional): update frequency. Defaults to 1.
        inner (Chainable | None, optional): inner modules. Defaults to None.

    """

    def __init__(
        self,
        rank: int = 100,
        beta=0.95,
        eig_tol: float | None = 1e-5,
        damping: float = 0,
        rdamping: float = 0,
        mm_tol: float = 0,
        mm_truncate: int | None = None,
        mm_damping: float = 1e-4,
        mm_rdamping: float = 0,
        id_reg: float | None = None,
        column_space_tol = 1e-9,

        orthogonalize_interval: int | None = None,
        orthogonalize_method: OrthogonalizeMethod = 'qr',

        eigenbasis_optimizer: LREOptimizerBase | None = None,
        concat_params: bool = True,
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        for k in ["self", "concat_params", "inner", "update_freq"]:
            del defaults[k]

        super().__init__(defaults, concat_params=concat_params, inner=inner, update_freq=update_freq)

    def single_tensor_update(self, tensor, param, grad, loss, state, setting):
        state["step"] = state.get("step", 0) + 1
        beta = setting["beta"]

        if "L" not in state:
            # for uu^T u is eigenvector and u^T u is eigenvalue
            norm = torch.linalg.vector_norm(tensor).clip(min=torch.finfo(tensor.dtype).tiny * 2) # pylint:disable=not-callable

            state["L"] = state["L_reg"] = (tensor.dot(tensor).unsqueeze(0) / norm) # (rank,)
            state["Q"] = state["Q_reg"] = tensor.unsqueeze(-1) / norm # (m, rank)

        else:
            try:
                L = state["L"]
                Q = state["Q"]

                # compute new factors
                L_new, Q_new = eigh_plus_uuT(L*beta, Q, tensor, alpha=(1-beta), tol=setting["column_space_tol"], retry_float64=True)

                # truncate/regularize new factors (those go into the accumulator)
                L_new, Q_new = regularize_eigh(L=L_new, Q=Q_new, truncate=setting["rank"], tol=setting["eig_tol"],
                                              damping=setting["damping"], rdamping=setting["rdamping"])

                _eigengrad_update_state_(state=state, setting=setting, L_new=L_new, Q_new=Q_new)

            except torch.linalg.LinAlgError:
                pass

    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):
        if "L_reg" not in state:
            return tensor.clip(-0.1, 0.1)

        if "eigenbasis_state" not in state:
            state["eigenbasis_state"] = {}

        return eigengrad_apply(
            tensor = tensor,
            L_reg = state["L_reg"],
            Q_reg = state["Q_reg"],
            beta = setting["beta"],
            step = state["step"],
            debias = True,
            id_reg = setting["id_reg"],
            eigenbasis_optimizer = setting["eigenbasis_optimizer"],
            eigenbasis_state = state["eigenbasis_state"]
        )
