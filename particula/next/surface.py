"""Strategies for curvature effects on particles. This includes
the Kelvin effect, and surface tension methods.

Should add the organic film strategy at some point"""

# pyright: reportArgumentType=false

from abc import ABC, abstractmethod
from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.convert import (
    mass_concentration_to_mole_fraction,
    mass_concentration_to_volume_fraction
    )
from particula.constants import GAS_CONSTANT


def kelvin_radius(
    effective_surface_tension: Union[float, NDArray[np.float_]],
    effective_density: Union[float, NDArray[np.float_]],
    molar_mass: Union[float, NDArray[np.float_]],
    temperature: float,
) -> Union[float, NDArray[np.float_]]:
    """
        Calculate the Kelvin radius which determines the curvature effect on
        vapor pressure.

        Args:
        -----
        - surface_tension (float or NDArray[float]): Surface tension of the
        mixture [N/m].
        - molar_mass (float or NDArray[float]): Molar mass of the species
        [kg/mol].
        - mass_concentration (float or NDArray[float]): Concentration of the
        species [kg/m^3].
        - temperature (float): Temperature of the system [K].

        Returns:
        --------
        - float or NDArray[float]: Kelvin radius [m].

        References:
        -----------
        - Based on Neil Donahue's approach to the Kelvin equation:
        r = 2 * surface_tension * molar_mass / (R * T * density)
        See more: https://en.wikipedia.org/wiki/Kelvin_equation
        """
    return (
        (2 * effective_surface_tension * molar_mass)
        / (GAS_CONSTANT.m * temperature * effective_density)
    )


def kelvin_term(
    radius: Union[float, NDArray[np.float_]],
    kelvin_radius_value: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]:
    """
    Calculate the Kelvin term, which quantifies the effect of particle
    curvature on vapor pressure.

    Args:
    -----
    - radius (float or NDArray[float]): Radius of the particle [m].
    - kelvin_radius (float or NDArray[float]): Kelvin radius [m].

    Returns:
    --------
    - float or NDArray[float]: The exponential factor adjusting vapor
    pressure due to curvature.

    References:
        Based on Neil Donahue's approach to the Kelvin equation:
        exp(kelvin_radius / particle_radius)
        See more: https://en.wikipedia.org/wiki/Kelvin_equation
    """
    return np.exp(kelvin_radius_value / radius)


class SurfaceStrategy(ABC):
    """
    Abstract class for implementing strategies to calculate surface tension and
    the Kelvin effect for species in particulate phases.
    """

    @abstractmethod
    def effective_surface_tension(
        self,
        mass_concentration: Union[float, NDArray[np.float_]]
    ) -> float:
        """
        Calculate the effective surface tension of species based on their
        concentration.

        Args:
        -----
        - mass_concentration (float or NDArray[float]): Concentration of the
        species [kg/m^3].

        Returns:
        -------
        - float or NDArray[float]: Effective surface tension [N/m].
        """

    @abstractmethod
    def effective_density(
        self,
        mass_concentration: Union[float, NDArray[np.float_]]
    ) -> float:
        """
        Calculate the effective density of species based on their
        concentration.

        Args:
        -----
        - mass_concentration (float or NDArray[float]): Concentration of the
        species [kg/m^3].

        Returns:
        --------
        - float or NDArray[float]: Effective density of the species [kg/m^3].
        """

    def kelvin_radius(
        self,
        molar_mass: Union[float, NDArray[np.float_]],
        mass_concentration: Union[float, NDArray[np.float_]],
        temperature: float
    ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the Kelvin radius which determines the curvature effect on
        vapor pressure.

        Args:
        -----
        - surface_tension (float or NDArray[float]): Surface tension of the
        mixture [N/m].
        - molar_mass (float or NDArray[float]): Molar mass of the species
        [kg/mol].
        - mass_concentration (float or NDArray[float]): Concentration of the
        species [kg/m^3].
        - temperature (float): Temperature of the system [K].

        Returns:
        --------
        - float or NDArray[float]: Kelvin radius [m].

        References:
        -----------
        - Based on Neil Donahue's approach to the Kelvin equation:
        r = 2 * surface_tension * molar_mass / (R * T * density)
        See more: https://en.wikipedia.org/wiki/Kelvin_equation
        """
        return kelvin_radius(
            self.effective_surface_tension(mass_concentration),
            self.effective_density(mass_concentration),
            molar_mass,
            temperature
        )

    def kelvin_term(
        self,
        radius: Union[float, NDArray[np.float_]],
        molar_mass: Union[float, NDArray[np.float_]],
        mass_concentration: Union[float, NDArray[np.float_]],
        temperature: float
    ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the Kelvin term, which quantifies the effect of particle
        curvature on vapor pressure.

        Args:
        -----
        - radius (float or NDArray[float]): Radius of the particle [m].
        - molar_mass (float or NDArray[float]): Molar mass of the species a
        [kg/mol].
        - mass_concentration (float or NDArray[float]): Concentration of the
        species [kg/m^3].
        - temperature (float): Temperature of the system [K].

        Returns:
        --------
        - float or NDArray[float]: The exponential factor adjusting vapor
        pressure due to curvature.

        References:
            Based on Neil Donahue's approach to the Kelvin equation:
            exp(kelvin_radius / particle_radius)
            See more: https://en.wikipedia.org/wiki/Kelvin_equation
        """
        return kelvin_term(
            radius,
            self.kelvin_radius(molar_mass, mass_concentration, temperature),
        )


# Surface mixing strategies
class MolarSurfaceStrategy(SurfaceStrategy):
    """Surface tension and density, based on mole fraction weighted values.

    Keyword arguments:
    ------------------
    - surface_tension (Union[float, NDArray[np.float_]]): Surface tension of
    the species [N/m]. If a single value is provided, it will be used for all
    species.
    - density (Union[float, NDArray[np.float_]]): Density of the species
    [kg/m^3]. If a single value is provided, it will be used for all species.
    - molar_mass (Union[float, NDArray[np.float_]]): Molar mass of the species
    [kg/mol]. If a single value is provided, it will be used for all species.

    References:
    -----------
    - Mole Fractions https://en.wikipedia.org/wiki/Mole_fraction
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float_]] = 0.072,  # water
        density: Union[float, NDArray[np.float_]] = 1000,  # water
        molar_mass: Union[float, NDArray[np.float_]] = 0.01815  # water
    ):
        self.surface_tension = surface_tension
        self.density = density
        self.molar_mass = molar_mass

    def effective_surface_tension(
                self,
                mass_concentration: Union[float, NDArray[np.float_]]
            ) -> float:
        if isinstance(self.surface_tension, float):
            return self.surface_tension
        return np.sum(
            self.surface_tension * mass_concentration_to_mole_fraction(
                mass_concentration, self.molar_mass),  # type: ignore
            dtype=np.float_)

    def effective_density(
                self,
                mass_concentration: Union[float, NDArray[np.float_]]
            ) -> float:
        if isinstance(self.density, float):
            return self.density
        return np.sum(
            self.density
            * mass_concentration_to_mole_fraction(
                mass_concentration, self.molar_mass),  # type: ignore
            dtype=np.float_
            )


class MassSurfaceStrategy(SurfaceStrategy):
    """Surface tension and density, based on mass fraction weighted values.

    Keyword arguments:
    ------------------
    - surface_tension (Union[float, NDArray[np.float_]]): Surface tension of
    the species [N/m]. If a single value is provided, it will be used for all
    species.
    - density (Union[float, NDArray[np.float_]]): Density of the species
    [kg/m^3]. If a single value is provided, it will be used for all species.

    References:
    -----------
    - Mass Fractions https://en.wikipedia.org/wiki/Mass_fraction_(chemistry)
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float_]] = 0.072,  # water
        density: Union[float, NDArray[np.float_]] = 1000  # water
    ):
        self.surface_tension = surface_tension
        self.density = density

    def effective_surface_tension(
                self,
                mass_concentration: Union[float, NDArray[np.float_]]
            ) -> float:
        if isinstance(self.surface_tension, float):
            return self.surface_tension
        return np.sum(
            self.surface_tension
            * mass_concentration
            / np.sum(mass_concentration),
            dtype=np.float_)

    def effective_density(
                self,
                mass_concentration: Union[float, NDArray[np.float_]]
            ) -> float:
        if isinstance(self.density, float):
            return self.density
        return np.sum(
            self.density
            * mass_concentration
            / np.sum(mass_concentration),
            dtype=np.float_
            )


class VolumeSurfaceStrategy(SurfaceStrategy):
    """Surface tension and density, based on volume fraction weighted values.

    Keyword arguments:
    ------------------
    - surface_tension (Union[float, NDArray[np.float_]]): Surface tension of
    the species [N/m]. If a single value is provided, it will be used for all
    species.
    - density (Union[float, NDArray[np.float_]]): Density of the species
    [kg/m^3]. If a single value is provided, it will be used for all species.

    References:
    -----------
    - Volume Fractions https://en.wikipedia.org/wiki/Volume_fraction
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float_]] = 0.072,  # water
        density: Union[float, NDArray[np.float_]] = 1000  # water
    ):
        self.surface_tension = surface_tension
        self.density = density

    def effective_surface_tension(
                self,
                mass_concentration: Union[float, NDArray[np.float_]]
            ) -> float:
        if isinstance(self.surface_tension, float):
            return self.surface_tension
        return np.sum(
            self.surface_tension
            * mass_concentration_to_volume_fraction(
                mass_concentration, self.density),  # type: ignore
            dtype=np.float_)

    def effective_density(
                self,
                mass_concentration: Union[float, NDArray[np.float_]]
            ) -> float:
        if isinstance(self.density, float):
            return self.density
        return np.sum(
            self.density
            * mass_concentration_to_volume_fraction(
                mass_concentration, self.density),  # type: ignore
            dtype=np.float_
            )


# Surface strategies factory
def surface_strategy_factory(
    strategy_type: str = "default",
    **kwargs: dict  # type: ignore
) -> SurfaceStrategy:
    """
    Factory function to create surface tension strategies.

    Args:
    -----
    - strategy_type (str): Type of strategy to create, options are:
        - "molar": Surface tension and density based on mole fractions.
        - "mass": Surface tension and density based on mass fractions.
        - "volume": Surface tension and density based on volume fractions.
    - **kwargs (dict): Keyword arguments for the strategy.

    Returns:
    --------
    - SurfaceStrategy: Instance of the surface tension strategy.
    """
    if strategy_type.lower() == "molar":
        return MolarSurfaceStrategy(**kwargs)
    if strategy_type.lower() == "mass":
        return MassSurfaceStrategy(**kwargs)
    if strategy_type.lower() == "volume":
        return VolumeSurfaceStrategy(**kwargs)
    if strategy_type.lower() == "default":
        return VolumeSurfaceStrategy()
    raise ValueError(
        f"Surface strategy type '{strategy_type}' not recognized.")
