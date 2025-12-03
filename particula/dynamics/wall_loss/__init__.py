"""Wall loss utilities and strategies.

This package provides wall loss rate functions and strategy classes
for particle wall deposition in various geometries.
"""

from .rate import (
    get_rectangle_wall_loss_rate as get_rectangle_wall_loss_rate,
)
from .rate import (
    get_spherical_wall_loss_rate as get_spherical_wall_loss_rate,
)
from .wall_loss_strategies import (
    SphericalWallLossStrategy as SphericalWallLossStrategy,
)
from .wall_loss_strategies import (
    WallLossStrategy as WallLossStrategy,
)
