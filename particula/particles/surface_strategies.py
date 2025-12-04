"""Strategies for surface effects on particles.

Provides classes for calculating effective surface tension and the
Kelvin effect for species in particulate phases. Future expansions
may include an organic film strategy.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from particula.particles.properties.convert_mass_concentration import (
    get_mass_fraction_from_mass,
    get_mole_fraction_from_mass,
    get_volume_fraction_from_mass,
)
from particula.particles.properties.kelvin_effect_module import (
    get_kelvin_radius,
    get_kelvin_term,
)


def _as_2d(array: NDArray[np.float64]) -> tuple[NDArray[np.float64], bool]:
    """Promote *array* to 2-D (n_bins, n_species).

    Returns:
    -------
    arr2d : the reshaped array
    was_1d : True if the original input was 1-D (hence single-bin)
    """
    if array.ndim == 1:
        return array[np.newaxis, :], True
    if array.ndim != 2:
        raise ValueError("`values` must be 1-D or 2-D")
    return array, False


def _interp_surface_tension(
    temperature: float,
    surface_tension_table: NDArray[np.float64],
    temperature_table: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]:
    """Interpolate *surface_tension_table* at *temperature*.
    Handles 1-D (single species) and 2-D (one column per species) tables.

    Returns a scalar (for 1-D) or 1-D array (for 2-D).
    """
    if surface_tension_table.ndim == 1:  # 1-D lookup
        return np.interp(temperature, temperature_table, surface_tension_table)

    if surface_tension_table.ndim == 2:  # 2-D lookup, per-species columns
        return np.array(
            [
                np.interp(
                    temperature,
                    temperature_table,
                    surface_tension_table[:, j],
                )
                for j in range(surface_tension_table.shape[1])
            ],
            dtype=np.float64,
        )

    raise ValueError("surface_tension_table must be 1-D or 2-D")


def _broadcast_weights(
    weights: NDArray[np.float64], target_shape: tuple[int, int]
) -> NDArray[np.float64]:
    """Return *weights* with shape exactly equal to *target_shape*
    (n_bins, n_species).  Accepts 1-D (n_species,) or 2-D inputs.
    """
    n_bins, n_species = target_shape
    if weights.ndim == 1:
        if weights.size != n_species:
            raise ValueError("weights size must equal number of species")
        return np.tile(weights, (n_bins, 1))

    if weights.ndim == 2:
        if weights.shape == (n_species, n_bins):
            weights = weights.T  # transposed input
        # (a) single-bin weights → broadcast to all bins
        if weights.shape == (1, n_species) and n_bins > 1:
            weights = np.tile(weights, (n_bins, 1))

        # (b) multi-bin weights with single-bin target → collapse by averaging
        elif n_bins == 1 and weights.shape[0] > 1:
            weights = weights.mean(axis=0, keepdims=True)

        # (c) after the above adjustments the shape must match
        if weights.shape != (n_bins, n_species):
            raise ValueError(
                f"weights must have shape ({n_bins}, {n_species}) "
                f"after broadcasting/collapsing, but has shape {weights.shape}"
            )
        return weights

    raise ValueError("weights must be 1-D or 2-D")


def _weighted_average_by_phase(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    phase_index: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Return an array where each element equals the phase-averaged *values*
    using *weights* as weighting factors.

    For every unique entry in `phase_index` the weighted-average is
    computed and broadcast back onto all members of that phase.

    Arguments:
        - values : Array of values, shape (n_bins, n_species) or (n_species,)
        - weights : Array of weights corresponding to each species, shape
          (n_species,) **or** (n_bins, n_species).  A 1-D array is broadcast
          across the n_bins axis.
        - phase_index : Array indicating the phase index for each species,
          shape (n_species,).

    Returns:
        - averaged : Array of averaged values, same shape as *values* input.
          If the weight sum for a bin/phase is zero the arithmetic mean is used.
    """
    # --- normalise shapes -------------------------------------------------
    values, return_1d = _as_2d(np.asarray(values, dtype=np.float64))
    weights = _broadcast_weights(
        np.asarray(weights, dtype=np.float64), values.shape
    )

    averaged = np.empty_like(values)

    # --- do weight-averaging phase-by-phase -------------------------------
    for phase in np.unique(phase_index):
        species_mask = phase_index == phase  # (n_species,)
        w_phase = weights[:, species_mask]  # (n_bins, n_sp_phase)
        w_sum = w_phase.sum(axis=1, keepdims=True)  # (n_bins, 1)

        # bins where ∑w ≠ 0  -> weighted average
        weighted_num = (values[:, species_mask] * w_phase).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            weighted_avg = np.where(
                w_sum.squeeze() != 0,
                weighted_num / w_sum.squeeze(),
                np.nan,  # placeholder
            )

        # bins where ∑w == 0  -> simple (un-weighted) mean
        if np.isnan(weighted_avg).any():
            unweighted_avg = values[:, species_mask].mean(axis=1)
            weighted_avg = np.where(
                np.isnan(weighted_avg), unweighted_avg, weighted_avg
            )

        # broadcast result back on all members of the current phase
        averaged[:, species_mask] = weighted_avg[:, np.newaxis]

    # --- restore caller's dimensionality ---------------------------------
    return averaged[0] if return_1d else averaged


class SurfaceStrategy(ABC):
    """Abstract base class for surface strategies.

    Implements methods for calculating effective surface tension, density,
    and the Kelvin effect in particulate phases.

    Methods:
    - effective_surface_tension : Calculate the effective surface tension.
    - effective_density : Calculate the effective density.
    - get_name : Return the type of the surface strategy.
    - kelvin_radius : Calculate the Kelvin radius for curvature effects.
    - kelvin_term : Calculate the exponential Kelvin term for vapor pressure.
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,  # water
        surface_tension_table: Optional[NDArray[np.float64]] = None,
        temperature_table: Optional[NDArray[np.float64]] = None,
    ):
        """Initialize the surface strategy."""
        self.surface_tension = np.asarray(surface_tension, dtype=np.float64)
        self.surface_tension_table = (
            np.asarray(surface_tension_table, dtype=np.float64)
            if surface_tension_table is not None
            else None
        )
        self.temperature_table = (
            np.asarray(temperature_table, dtype=np.float64)
            if temperature_table is not None
            else None
        )

    @abstractmethod
    def effective_surface_tension(
        self,
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the effective surface tension of the species mixture.

        Arguments:
            - mass_concentration : Concentration of the species in kg/m^3.
            - temperature : Optional temperature dependence in K.

        Returns:
            - Effective surface tension in N/m.
        """

    @abstractmethod
    def get_density(self) -> Union[float, NDArray[np.float64]]:
        """Get density of the species mixture.

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
        """Calculate the Kelvin radius, which sets the curvature effect on vapor
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
            self.effective_surface_tension(mass_concentration, temperature),
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
        """Calculate the exponential Kelvin term that adjusts vapor pressure.

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

    def update_surface_tension(self, temperature: float) -> None:
        """Update the surface tension attribute based on the given temperature.

        If a temperature-dependent surface tension table is provided,
        it interpolates the surface tension value for the specified
        temperature.

        Arguments:
            - temperature : Temperature in K to update the surface tension.
        """
        if self.surface_tension_table is None or self.temperature_table is None:
            return

        self.surface_tension = np.asarray(
            _interp_surface_tension(
                temperature,
                self.surface_tension_table,
                self.temperature_table,
            ),
            dtype=np.float64,
        )


# Surface mixing strategies
class SurfaceStrategyMolar(SurfaceStrategy):
    """Surface tension and density based on mole-fraction weighting.

    Attributes:
        - surface_tension : Surface tension array or scalar in N/m.
        - density : Density array or scalar in kg/m^3.
        - molar_mass : Molar mass array or scalar in kg/mol.
        - phase_index : Optional array indicating phase indices for species.
          For example, [0, 1, 1] for two phases, where the first species
          belongs to phase 0 and the next two to phase 1.
        - surface_tension_table : Optional 2D array for temperature-dependent
          surface tension values.
        - temperature_table : Optional 1D array for temperature values
          corresponding to the surface tension table.

    References:
        - [Mole Fraction](https://en.wikipedia.org/wiki/Mole_fraction)
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,  # water
        density: Union[float, NDArray[np.float64]] = 1000,  # water
        molar_mass: Union[float, NDArray[np.float64]] = 0.01815,  # water
        phase_index: Optional[Union[Sequence[int], NDArray[np.int_]]] = None,
        surface_tension_table: Optional[NDArray[np.float64]] = None,
        temperature_table: Optional[NDArray[np.float64]] = None,
    ):
        """Initialize the SurfaceStrategyMolar.

        Sets up the molar-fraction-based surface strategy with surface
        tension, density, and molar mass parameters for calculating
        effective surface properties.
        """
        super().__init__(
            surface_tension=surface_tension,
            surface_tension_table=surface_tension_table,
            temperature_table=temperature_table,
        )
        self.density = density
        self.molar_mass = molar_mass
        self.phase_index = (
            None if phase_index is None else np.array(phase_index, dtype=int)
        )

    def effective_surface_tension(
        self,
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate effective surface tension using mole-fraction weighting.

        Returns:
            Effective surface tension in N/m.
        """
        if temperature is not None:
            self.update_surface_tension(temperature)

        if isinstance(self.surface_tension, float) or self.phase_index is None:
            return np.asarray(self.surface_tension, dtype=np.float64)

        mole_frac = get_mole_fraction_from_mass(
            mass_concentration,
            self.molar_mass,  # type: ignore
        )
        return _weighted_average_by_phase(
            np.asarray(self.surface_tension, dtype=np.float64),
            mole_frac,
            self.phase_index,
        )

    def get_density(self) -> Union[float, NDArray[np.float64]]:
        """Get density of the species mixture.

        Returns:
            Density in kg/m^3.
        """
        return self.density


class SurfaceStrategyMass(SurfaceStrategy):
    """Surface tension and density based on mass-fraction weighting.

    Attributes:
        - surface_tension : Surface tension array or scalar in N/m.
        - density : Density array or scalar in kg/m^3.
        - phase_index : Optional array indicating phase indices for species.
          Example: [0, 1, 1] → three species in two phases (first in phase 0,
          last two in phase 1).

    References:
    - [Mass Fraction](https://en.wikipedia.org/wiki/Mass_fraction_(chemistry))
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,  # water
        density: Union[float, NDArray[np.float64]] = 1000,  # water
        phase_index: Optional[Union[Sequence[int], NDArray[np.int_]]] = None,
        surface_tension_table: Optional[NDArray[np.float64]] = None,
        temperature_table: Optional[NDArray[np.float64]] = None,
    ):
        """Initialize the SurfaceStrategyMass.

        Sets up the mass-fraction-based surface strategy with surface
        tension and density parameters for calculating effective surface
        properties.
        """
        super().__init__(
            surface_tension=surface_tension,
            surface_tension_table=surface_tension_table,
            temperature_table=temperature_table,
        )
        self.density = density
        self.phase_index = (
            None if phase_index is None else np.array(phase_index, dtype=int)
        )

    def effective_surface_tension(
        self,
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate effective surface tension using mass-fraction weighting.

        Returns:
            Effective surface tension in N/m.
        """
        if temperature is not None:
            self.update_surface_tension(temperature)

        if isinstance(self.surface_tension, float) or self.phase_index is None:
            return np.asarray(self.surface_tension, dtype=np.float64)

        mass_fraction = get_mass_fraction_from_mass(mass_concentration)
        return _weighted_average_by_phase(
            np.asarray(self.surface_tension, dtype=np.float64),
            mass_fraction,
            self.phase_index,
        )

    def get_density(self) -> Union[float, NDArray[np.float64]]:
        """Get density of the species mixture.

        Returns:
            Density in kg/m^3.
        """
        return self.density


class SurfaceStrategyVolume(SurfaceStrategy):
    """Surface tension and density based on volume-fraction weighting.

    Attributes:
        - surface_tension : Surface tension array or scalar in N/m.
        - density : Density array or scalar in kg/m^3.
        - phase_index : Optional array indicating phase indices for species.
          Example: [0, 1, 1] → three species in two phases (first in phase 0,
          last two in phase 1).

    References:
        - [Volume Fraction](https://en.wikipedia.org/wiki/Volume_fraction)
    """

    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,  # water
        density: Union[float, NDArray[np.float64]] = 1000,  # water
        phase_index: Optional[Union[Sequence[int], NDArray[np.int_]]] = None,
        surface_tension_table: Optional[NDArray[np.float64]] = None,
        temperature_table: Optional[NDArray[np.float64]] = None,
    ):
        """Initialize the SurfaceStrategyVolume.

        Sets up the volume-fraction-based surface strategy with surface
        tension and density parameters for calculating effective surface
        properties.
        """
        super().__init__(
            surface_tension=surface_tension,
            surface_tension_table=surface_tension_table,
            temperature_table=temperature_table,
        )
        self.density = density
        self.phase_index = (
            None if phase_index is None else np.array(phase_index, dtype=int)
        )

    def effective_surface_tension(
        self,
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate effective surface tension using volume-fraction weighting.

        Returns:
            Effective surface tension in N/m.
        """
        if temperature is not None:
            self.update_surface_tension(temperature)

        if isinstance(self.surface_tension, float) or self.phase_index is None:
            return np.asarray(self.surface_tension, dtype=np.float64)

        vol_frac = get_volume_fraction_from_mass(
            mass_concentration,
            self.density,  # type: ignore
        )
        return _weighted_average_by_phase(
            np.asarray(self.surface_tension, dtype=np.float64),
            vol_frac,
            self.phase_index,
        )

    def get_density(self) -> Union[float, NDArray[np.float64]]:
        """Get density of the species mixture.

        Returns:
            Density in kg/m^3.
        """
        return self.density
