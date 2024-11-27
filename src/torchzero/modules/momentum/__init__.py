"""
Modules that implement momentum.
"""
from .polyak_momentum import PolyakMomentum
from .nesterov_momentum import NesterovMomentum
from .random_coordinate_momentum import RandomCoordinateMomentum, RandomCoordinateNesterovMomentum
from .gradient_averaging import GradientAveraging