"""CPU and CUDA activities for the pinned AIOMFAC-web model."""

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

__all__ = [
    "AIOMFACModel",
    "ActivityResult",
    "ConvergenceError",
    "MixtureSpec",
    "SpeciationRequiredError",
]
