"""Synthetic materials for multi-reservoir forecasts."""

from dataclasses import dataclass

import numpy as np
from dual_materials import (
    ACCOMMODATION,
    DENSITY,
    DIFFUSIVITY,
    MOLAR,
    NAMES,
    PURE_PRESSURE,
    TEMPERATURE,
    R,
)


@dataclass(frozen=True)
class Materials:
    """The gas lanes exchange with the first particle materials in order."""

    names: tuple[str, ...]
    molar: np.ndarray
    density: np.ndarray
    pressure: np.ndarray
    diffusivity: np.ndarray
    accommodation: np.ndarray

    @property
    def species(self):
        """Number of volatile species."""
        return len(self.names)

    @property
    def saturation(self):
        """Saturated gas mass concentrations in kg per cubic metre."""
        return self.pressure * self.molar[: self.species] / (R * TEMPERATURE)


def materials(count):
    """Add synthetic species with distinct vapor pressure and response time."""
    if count not in (1, 2, 3, 4):
        raise ValueError("Use one through four volatile species")
    return Materials(
        (*NAMES, "organic_B", "organic_C")[:count],
        np.r_[np.r_[MOLAR[:2], 0.085, 0.165][:count], MOLAR[2:]],
        np.r_[np.r_[DENSITY[:2], 1050.0, 1350.0][:count], DENSITY[2:]],
        np.r_[PURE_PRESSURE, 58.0, 2.4][:count],
        np.r_[DIFFUSIVITY, 9.2e-6, 5.4e-6][:count],
        np.r_[ACCOMMODATION, 0.29, 0.10][:count],
    )
