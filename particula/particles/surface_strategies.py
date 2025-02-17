"""Strategies for surface effects on particles.

Should add the organic film strategy in the future.
"""

from abc import ABC, abstractmethod
from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.convert import (
    mass_concentration_to_mole_fraction,
    mass_concentration_to_volume_fraction,
)
from particula.particles.properties import kelvin_radius, kelvin_term


class SurfaceStrategy(ABC):
    """ABC class for Surface Strategies.

    Abstract class for implementing strategies to calculate surface tension
    and the Kelvin effect for species in particulate phases.

    Methods:
        effective_surface_tension: Calculate the effective surface tension of
            species based on their concentration.
        effective_density: Calculate the effective density of species based on
            their concentration.
        get_name: Return the type of the surface strategy.
        kelvin_radius: Calculate the Kelvin radius which determines the
            curvature effect on vapor pressure.
        kelvin_term: Calculate the Kelvin term, which quantifies the effect of
            particle curvature on vapor pressure.
    """

    @abstractmethod
    def effective_surface_tension(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        """Calculate the effective surface tension of the species mixture.

        Args:
            mass_concentration: Concentration of the species [kg/m^3].

        Returns:
            float or NDArray[float]: Effective surface tension [N/m].
        """

    @abstractmethod
    def effective_density(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        """Calculate the effective density of the species mixture.

        Args:
            mass_concentration: Concentration of the species [kg/m^3].

        Returns:
            float or NDArray[float]: Effective density of the species [kg/m^3].
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
        """Calculate the Kelvin radius which determines the curvature effect.

        The kelvin radius is molecule specific and depends on the surface
        tension, molar mass, density, and temperature of the system. It is
        used to calculate the Kelvin term, which quantifies the effect of
        particle curvature on vapor pressure.

        Args:
            surface_tension: Surface tension of the mixture [N/m].
            molar_mass: Molar mass of the species [kg/mol].
            mass_concentration: Concentration of the species [kg/m^3].
            temperature: Temperature of the system [K].

        Returns:
            float or NDArray[float]: Kelvin radius [m].

        References:
        - Based on Neil Donahue's approach to the Kelvin equation:
        r = 2 * surface_tension * molar_mass / (R * T * density)
        [Kelvin Wikipedia](https://en.wikipedia.org/wiki/Kelvin_equation)
        """
        return kelvin_radius(
            self.effective_surface_tension(mass_concentration),
            self.effective_density(mass_concentration),
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
        """Calculate the Kelvin term, which multiplies the vapor pressure.

        The Kelvin term is used to adjust the vapor pressure of a species due
        to the curvature of the particle.

        Args:
            radius: Radius of the particle [m].
            molar_mass: Molar mass of the species a [kg/mol].
            mass_concentration: Concentration of the species [kg/m^3].
            temperature: Temperature of the system [K].

        Returns:
            float or NDArray[float]: The exponential factor adjusting vapor
                pressure due to curvature.

        References:
        Based on Neil Donahue's approach to the Kelvin equation:
        exp(kelvin_radius / particle_radius)
        [Kelvin Eq Wikipedia](https://en.wikipedia.org/wiki/Kelvin_equation)
        """
        return kelvin_term(
            radius,
            self.kelvin_radius(molar_mass, mass_concentration, temperature),
        )


# Surface mixing strategies
class SurfaceStrategyMolar(SurfaceStrategy):
    """Surface tension and density, based on mole fraction weighted values.

    Arguments:
        surface_tension: Surface tension of the species [N/m]. If a single
            value is provided, it will be used for all species.
        density: Density of the species [kg/m^3]. If a single value is
            provided, it will be used for all species.
        molar_mass: Molar mass of the species [kg/mol]. If a single value is
            provided, it will be used for all species.

    References:
        [Mole Fractions](https://en.wikipedia.org/wiki/Mole_fraction)
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,  # water
        density: Union[float, NDArray[np.float64]] = 1000,  # water
        molar_mass: Union[float, NDArray[np.float64]] = 0.01815,  # water
    ):
        self.surface_tension = surface_tension
        self.density = density
        self.molar_mass = molar_mass

    def effective_surface_tension(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        if isinstance(self.surface_tension, float):
            return self.surface_tension
        return np.sum(
            self.surface_tension
            * mass_concentration_to_mole_fraction(
                mass_concentration, self.molar_mass  # type: ignore
            ),
            dtype=np.float64,
        )

    def effective_density(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        if isinstance(self.density, float):
            return self.density
        return np.sum(
            self.density
            * mass_concentration_to_mole_fraction(
                mass_concentration, self.molar_mass  # type: ignore
            ),
            dtype=np.float64,
        )


class SurfaceStrategyMass(SurfaceStrategy):
    """Surface tension and density, based on mass fraction weighted values.

    Args:
        surface_tension: Surface tension of the species [N/m]. If a single
            value is provided, it will be used for all species.
        density: Density of the species [kg/m^3]. If a single value is
            provided, it will be used for all species.

    References:
    [Mass Fractions](https://en.wikipedia.org/wiki/Mass_fraction_(chemistry))
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,  # water
        density: Union[float, NDArray[np.float64]] = 1000,  # water
    ):
        self.surface_tension = surface_tension
        self.density = density

    def effective_surface_tension(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        if isinstance(self.surface_tension, float):
            return self.surface_tension
        return np.sum(
            self.surface_tension
            * mass_concentration
            / np.sum(mass_concentration),
            dtype=np.float64,
        )

    def effective_density(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        if isinstance(self.density, float):
            return self.density
        return np.sum(
            self.density * mass_concentration / np.sum(mass_concentration),
            dtype=np.float64,
        )


class SurfaceStrategyVolume(SurfaceStrategy):
    """Surface tension and density, based on volume fraction weighted values.

    Args:
        surface_tension: Surface tension of the species [N/m]. If a single
            value is provided, it will be used for all species.
        density: Density of the species [kg/m^3]. If a single value is
            provided, it will be used for all species.

    References:
    [Volume Fractions](https://en.wikipedia.org/wiki/Volume_fraction)
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,  # water
        density: Union[float, NDArray[np.float64]] = 1000,  # water
    ):
        self.surface_tension = surface_tension
        self.density = density

    def effective_surface_tension(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        if isinstance(self.surface_tension, float):
            return self.surface_tension
        return np.sum(
            self.surface_tension
            * mass_concentration_to_volume_fraction(
                mass_concentration, self.density  # type: ignore
            ),
            dtype=np.float64,
        )

    def effective_density(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> float:
        if isinstance(self.density, float):
            return self.density
        return np.sum(
            self.density
            * mass_concentration_to_volume_fraction(
                mass_concentration, self.density  # type: ignore
            ),
            dtype=np.float64,
        )
