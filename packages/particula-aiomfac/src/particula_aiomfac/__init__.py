"""CPU and CUDA activities for the pinned AIOMFAC-web model."""

# The backend imports these pinned reference constants during initialization.
# ruff: noqa: E402
__version__ = "0.1.4"
UPSTREAM_VERSION = "3.15"
UPSTREAM_REVISION = "24041a1b0db090de1285008814a5285b4b9b746b"

from .model import (
    ActivityResult,
    AIOMFACModel,
    ConvergenceError,
    MixtureSpec,
    SpeciationRequiredError,
)
from .native_inventory import SpeciesLayout
from .reservoirs import (
    GasReaction,
    IonSurface,
    SpeciesEquilibrium,
    SpeciesInventory,
)
from .solids import (
    SolidLiquidEquilibrium,
    SolidSalt,
    conserved_species_matrix,
    solve_ionic_solid_liquid,
    solve_solid_liquid,
)

__all__ = [
    "GasReaction",
    "IonSurface",
    "SpeciesEquilibrium",
    "SpeciesInventory",
    "SpeciesLayout",
    "AIOMFACModel",
    "ActivityResult",
    "ConvergenceError",
    "MixtureSpec",
    "SpeciationRequiredError",
    "SolidLiquidEquilibrium",
    "SolidSalt",
    "conserved_species_matrix",
    "solve_ionic_solid_liquid",
    "solve_solid_liquid",
]
