"""Explicit Warp optical kernels and prepared numerical workspaces.

Importing CPU optics does not load this namespace or initialize Warp.
"""

from particula.optics.gpu.bulk import (
    PopulationWorkspace,
    prepare_population_mie,
)
from particula.optics.gpu.shortwave import ShortwaveWorkspace, prepare_shortwave
from particula.optics.gpu.workspace import MieWorkspace, prepare_mie

__all__ = [
    "ShortwaveWorkspace",
    "prepare_shortwave",
    "MieWorkspace",
    "prepare_mie",
    "PopulationWorkspace",
    "prepare_population_mie",
]
