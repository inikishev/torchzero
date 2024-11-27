r"""
Modules that project the parameters into lower dimensional space,
making Newton-like methods feasible for large-scale problems.
"""
from .random_subspace import Subspace, ProjRandom, ProjAscent, ProjAscentRay, ProjGrad, ProjGradRay, Proj2Masks
