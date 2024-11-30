r"""
Modules that project the parameters into lower dimensional space,
making Newton-like methods feasible for large-scale problems.
"""
from .random_subspace import (Proj2Masks, ProjAscent, ProjAscentRay, ProjGrad,
                              ProjGradAscentDifference, ProjGradRay,
                              ProjLastAscentDifference, ProjLastGradDifference,
                              ProjNormalize, ProjRandom, Subspace)
