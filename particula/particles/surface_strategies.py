"""
Strategies for surface effects on particles.

Provides classes for calculating effective surface tension and the
Kelvin effect for species in particulate phases. Future expansions
may include an organic film strategy.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Optional
from numpy.typing import NDArray
import numpy as np

from particula.particles.properties.convert_mass_concentration import (
    get_mole_fraction_from_mass,
    get_volume_fraction_from_mass,
    get_mass_fraction_from_mass,
)
from particula.particles.properties.kelvin_effect_module import (
    get_kelvin_radius,
    get_kelvin_term,
)


def _weighted_average_by_phase(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    phase_index: NDArray[np.int_],
) -> NDArray[np.float64]:
    """
    Return an array where each element equals the phase-averaged *values*
    using *weights* as weighting factors.

    For every unique entry in `phase_index` the weighted-average is
    computed and broadcast back onto all members of that phase.

    Arguments:
        - values : Array of values to be averaged, shape (n_species,).
        - weights : Array of weights corresponding to each species, shape
          (n_species,).
        - phase_index : Array indicating the phase index for each species,
          shape (n_species,).

    Returns:
        - averaged : Array of averaged values, shape (n_species,).
    """
    averaged = np.zeros_like(values, dtype=np.float64)
    for ph in np.unique(phase_index):
        mask = phase_index == ph
        if weights[mask].sum() != 0:
            phase_weights = weights[mask] / weights[mask].sum()
        else:
            phase_weights = weights[mask]
        averaged_value = np.sum(values[mask] * phase_weights, dtype=np.float64)
        averaged[mask] = averaged_value
    return averaged


class SurfaceStrategy(ABC):
    """
    Abstract base class for surface strategies.

    Implements methods for calculating effective surface tension, density,
    and the Kelvin effect in particulate phases.

    Methods:
    - effective_surface_tension : Calculate the effective surface tension.
    - effective_density : Calculate the effective density.
    - get_name : Return the type of the surface strategy.
    - kelvin_radius : Calculate the Kelvin radius for curvature effects.
    - kelvin_term : Calculate the exponential Kelvin term for vapor pressure.
    """

    @abstractmethod
    def effective_surface_tension(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the effective surface tension of the species mixture.

        Arguments:
            - mass_concentration : Concentration of the species in kg/m^3.

        Returns:
            - Effective surface tension in N/m.
        """

    @abstractmethod
    def get_density(
        self
    ) -> Union[float, NDArray[np.float64]]:
        """
        Get density of the species mixture.

        Returns:
            - density in kg/m^3.
        """

    def get_name(self) -> str:
        """Return the type of the surface strategy."""
        return self.__class__.__name__

    def kelvin_radius(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the Kelvin radius, which sets the curvature effect on vapor
        pressure.

        Arguments:
            - molar_mass : Molar mass of the species in kg/mol.
            - mass_concentration : Concentration of the species in kg/m^3.
            - temperature : Temperature of the system in K.

        Returns:
            - Kelvin radius in meters.

        References:
            - r = 2 × surface_tension × molar_mass / (R × T × density)
              [Kelvin Equation](https://en.wikipedia.org/wiki/Kelvin_equation)
        """
        return get_kelvin_radius(
            self.effective_surface_tension(mass_concentration),
            self.get_density(),
            molar_mass,
            temperature,
        )

    def kelvin_term(
        self,
        radius: Union[float, NDArray[np.float64]],
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the exponential Kelvin term that adjusts vapor pressure.

        Arguments:
            - radius : Particle radius in meters.
            - molar_mass : Molar mass of the species in kg/mol.
            - mass_concentration : Concentration of the species in kg/m^3.
            - temperature : Temperature of the system in K.

        Returns:
            - Factor by which vapor pressure is increased.

        References:
            - P_eff = P_sat × exp(kelvin_radius / particle_radius)
              [Kelvin Equation](https://en.wikipedia.org/wiki/Kelvin_equation)
        """
        return get_kelvin_term(
            radius,
            self.kelvin_radius(molar_mass, mass_concentration, temperature),
        )


# Surface mixing strategies
class SurfaceStrategyMolar(SurfaceStrategy):
    """
    Surface tension and density based on mole-fraction weighting.

    Attributes:
        - surface_tension : Surface tension array or scalar in N/m.
        - density : Density array or scalar in kg/m^3.
        - molar_mass : Molar mass array or scalar in kg/mol.
        - phase_index : Optional array indicating phase indices for species.
          For example, [0, 1, 1] for two phases, where the first species
          belongs to phase 0 and the next two to phase 1.

    References:
        - [Mole Fraction](https://en.wikipedia.org/wiki/Mole_fraction)
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,  # water
        density: Union[float, NDArray[np.float64]] = 1000,  # water
        molar_mass: Union[float, NDArray[np.float64]] = 0.01815,  # water
        phase_index: Optional[Union[Sequence[int], NDArray[np.int_]]] = None,
    ):
        self.surface_tension = surface_tension
        self.density = density
        self.molar_mass = molar_mass
        self.phase_index = (
            None if phase_index is None else np.array(phase_index, dtype=int)
        )

    def effective_surface_tension(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        if isinstance(self.surface_tension, float) or self.phase_index is None:
            # If surface tension is a scalar or no phase index is provided,
            # return it directly.
            return np.asarray(self.surface_tension, dtype=np.float64)

        mole_frac = get_mole_fraction_from_mass(
            mass_concentration, self.molar_mass  # type: ignore
        )
        return _weighted_average_by_phase(
            np.asarray(self.surface_tension, dtype=np.float64),
            mole_frac,
            self.phase_index,
        )

    def get_density(
        self
    ) -> Union[float, NDArray[np.float64]]:
        return self.density


class SurfaceStrategyMass(SurfaceStrategy):
    """
    Surface tension and density based on mass-fraction weighting.

    Attributes:
        - surface_tension : Surface tension array or scalar in N/m.
        - density : Density array or scalar in kg/m^3.

    References:
    - [Mass Fraction](https://en.wikipedia.org/wiki/Mass_fraction_(chemistry))
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,  # water
        density: Union[float, NDArray[np.float64]] = 1000,  # water
        phase_index: Optional[Union[Sequence[int], NDArray[np.int_]]] = None,
    ):
        self.surface_tension = surface_tension
        self.density = density
        self.phase_index = (
            None if phase_index is None else np.array(phase_index, dtype=int)
        )

    def effective_surface_tension(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        # If a single surface-tension value is supplied **or** no phase
        # information is given, just return the (possibly vector) value
        # unchanged – same rule as SurfaceStrategyMolar.
        if isinstance(self.surface_tension, float) or self.phase_index is None:
            return np.asarray(self.surface_tension, dtype=np.float64)

        mass_fraction = get_mass_fraction_from_mass(mass_concentration)
        return _weighted_average_by_phase(
            np.asarray(self.surface_tension, dtype=np.float64),
            mass_fraction,
            self.phase_index,
        )

    def get_density(
        self
    ) -> Union[float, NDArray[np.float64]]:
        return self.density


class SurfaceStrategyVolume(SurfaceStrategy):
    """
    Surface tension and density based on volume-fraction weighting.

    Attributes:
        - surface_tension : Surface tension array or scalar in N/m.
        - density : Density array or scalar in kg/m^3.

    References:
        - [Volume Fraction](https://en.wikipedia.org/wiki/Volume_fraction)
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,  # water
        density: Union[float, NDArray[np.float64]] = 1000,  # water
        phase_index: Optional[Union[Sequence[int], NDArray[np.int_]]] = None,
    ):
        self.surface_tension = surface_tension
        self.density = density
        self.phase_index = (
            None if phase_index is None else np.array(phase_index, dtype=int)
        )

    def effective_surface_tension(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        if isinstance(self.surface_tension, float) or self.phase_index is None:
            return np.asarray(self.surface_tension, dtype=np.float64)

        vol_frac = get_volume_fraction_from_mass(
            mass_concentration, self.density  # type: ignore
        )
        return _weighted_average_by_phase(
            np.asarray(self.surface_tension, dtype=np.float64),
            vol_frac,
            self.phase_index,
        )

    def get_density(
        self
    ) -> Union[float, NDArray[np.float64]]:
        return self.density
