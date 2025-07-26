from .averaging import Averaging, MedianAveraging, WeightedAveraging
from .cautious import (
    Cautious,
    IntermoduleCautious,
    ScaleByGradCosineSimilarity,
    ScaleModulesByCosineSimilarity,
    UpdateGradientSignConsistency,
)

from .momentum import NAG, HeavyBall, EMA
