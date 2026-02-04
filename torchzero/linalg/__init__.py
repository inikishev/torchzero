from . import linear_operator

from .matrix_power import (
    matrix_power_eigh,
    matrix_power_svd,
    MatrixPowerMethod,
)
from .orthogonalize import zeropower_via_eigh, zeropower_via_newtonschulz5, zeropower_via_svd, orthogonalize,OrthogonalizeMethod
from .qr import qr_householder
from .solve import cg, nystrom_sketch_and_solve, nystrom_pcg
from .eigh import nystrom_approximation, regularize_eigh
