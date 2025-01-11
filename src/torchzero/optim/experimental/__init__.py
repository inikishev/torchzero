"""Optimizers that I haven't tested and various (mostly stupid) ideas go there.
If something works well I will move it outside of experimental folder.
Otherwise all optimizers in this category should be considered unlikely to good for most tasks."""
from .experimental import (
    HVPDiagNewton,
    ExaggeratedNesterov,
    ExtraCautiousAdam,
    GradMin,
    InwardSGD,
    MinibatchRprop,
    MomentumDenominator,
    MomentumNumerator,
    MultistepSGD,
    RandomCoordinateMomentum,
    ReciprocalSGD,
)


from .ray_search import NewtonFDMRaySearch, LBFGSRaySearch