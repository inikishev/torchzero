from .averaging import Averaging, MedianAveraging, WeightedAveraging
from .cautious import (
    Cautious,
    IntermoduleCautious,
    ScaleByGradCosineSimilarity,
    ScaleModulesByCosineSimilarity,
    UpdateGradientSignConsistency,
)
from .ema import EMA, Debias, Debias2, EMASquared, SqrtEMASquared, CenteredEMASquared, CenteredSqrtEMASquared
from .experimental import CoordinateMomentum
# from .matrix_momentum import MatrixMomentum

from .momentum import NAG, HeavyBall
from .matrix_momentum import MatrixMomentum, AdaptiveMatrixMomentum
