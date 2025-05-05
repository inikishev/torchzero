from .averaging import Averaging, MedianAveraging, WeightedAveraging
from .cautious import (
    Cautious,
    IntermoduleCautious,
    ScaleByGradCosineSimilarity,
    ScaleModulesByCosineSimilarity,
    UpdateGradientSignConsistency,
)
from .ema import EMA, Debias1, Debias2, EMASquared, SqrtEMASquared
from .experimental import CoordinateMomentum
# from .matrix_momentum import MatrixMomentum

from .momentum import NAG, HeavyBall
