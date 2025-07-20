from . import linear_operator
from .matrix_funcs import (
    eigvals_func,
    inv_sqrt_2x2,
    matrix_power_eigh,
    singular_vals_func,
    x_inv,
)
from .orthogonalize import gram_schmidt
from .qr import qr_householder
from .solve import cg, nystrom_approximation, nystrom_sketch_and_solve
from .svd import randomized_svd
