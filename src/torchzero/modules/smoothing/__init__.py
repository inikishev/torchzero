r"""
Gradient smoothing and orthogonalization methods.
"""
from .laplacian_smoothing import LaplacianSmoothing, vector_laplacian_smoothing, gradient_laplacian_smoothing_
from .gaussian_smoothing import ApproxGaussianSmoothing
from .orthogonalization import Orthogonalize, orthogonalize_grad_
from .newtonschulz import ZeropowerViaNewtonSchulz, zeropower_via_newtonschulz_