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


def _as_2d(array: NDArray[np.float64]) -> tuple[NDArray[np.float64], bool]:
    """
    Promote *array* to 2-D (n_bins, n_species).

    Returns
    -------
    arr2d : the reshaped array
    was_1d : True if the original input was 1-D (hence single-bin)
    """
    if array.ndim == 1:
        return array[np.newaxis, :], True
    if array.ndim != 2:
        raise ValueError("`values` must be 1-D or 2-D")
    return array, False


def _broadcast_weights(
    weights: NDArray[np.float64], target_shape: tuple[int, int]
) -> NDArray[np.float64]:
    """
    Return *weights* with shape exactly equal to *target_shape*
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
    """
    Return an array where each element equals the phase-averaged *values*
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
    
    def __init__(
        self,
        temperature: Optional[float] = None

        temperature: Optional[float] = None,
    ):
        super().__init__(temperature=temperature)
        """
        Initialize the surface strategy.
        This base requires optional temperature.
        """
        self.temperature: Optional[float] = None

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
    def get_density(self) -> Union[float, NDArray[np.float64]]:
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
    
    def set_temperature(self, temperature: float) -> None:
        """
        Set the temperature for the surface strategy.

        Arguments:
            - temperature : Temperature in Kelvin.
        """
        self.temperature = temperature


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
        temperature: Optional[float] = None,
    ):
        super().__init__(temperature=temperature)
        super().__init__(temperature=temperature)
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

    def get_density(self) -> Union[float, NDArray[np.float64]]:
        return self.density


class SurfaceStrategyMass(SurfaceStrategy):
    """
    Surface tension and density based on mass-fraction weighting.

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
        temperature: Optional[float] = None,
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

    def get_density(self) -> Union[float, NDArray[np.float64]]:
        return self.density


class SurfaceStrategyVolume(SurfaceStrategy):
    """
    Surface tension and density based on volume-fraction weighting.

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

    def get_density(self) -> Union[float, NDArray[np.float64]]:
        return self.density


class SurfaceStrategyTemperatureMolar(SurfaceStrategy):
    """
    Temperature-dependent surface tension via DIPPR-106.
    Surface tension is calculated as:

    - σ(T) = A \times (1 - θ)^n \times (1 + Bθ + Cθ² + Dθ³)
        - σ is the surface tension in newtons per metre,
        - θ = T / T_c is the reduced temperature,
        - A, B, C, D and n are DIPPR-106 parameters.

    Arguments:
        - dippr_a : Parameter ``A`` in newtons per metre.
        - critical_temperature : Critical temperature ``T_c`` in kelvin.
        - dippr_b : Parameter ``B`` of the correlation.
        - dippr_c : Parameter ``C`` of the correlation.
        - dippr_d : Parameter ``D`` of the correlation.
        - dippr_n : Exponent ``n``.
        - density : Liquid density in kg/m³.
        - molar_mass : Molar mass in kg/mol.
        - temperature : Reference temperature in kelvin.
        - phase_index : Optional array mapping species to phases.

    Examples:
        ``` py title="Example"
        strat = SurfaceStrategyTemperatureMolar(0.072, 647.1)
        strat.effective_surface_tension(100.0, 298.0)
        ```

    References:
        - "DIPPR Project 801", AIChE.
    """

    def __init__(
        self,
        dippr_a: Union[float, NDArray[np.float64]],
        critical_temperature: Union[float, NDArray[np.float64]],
        dippr_b: Union[float, NDArray[np.float64]] = 0.0,
        dippr_c: Union[float, NDArray[np.float64]] = 0.0,
        dippr_d: Union[float, NDArray[np.float64]] = 0.0,
        dippr_n: Union[float, NDArray[np.float64]] = 1.256,
        density: Union[float, NDArray[np.float64]] = 1000,
        molar_mass: Union[float, NDArray[np.float64]] = 0.01815,
        temperature: float = 298.0,
        phase_index: Optional[Union[Sequence[int], NDArray[np.int_]]] = None,
    ):
        """Create a temperature-aware surface strategy.

        Arguments:
            - dippr_a : Parameter ``A`` in newtons per metre.
            - critical_temperature : Critical temperature ``T_c`` in kelvin.
            - dippr_b : Parameter ``B`` of the correlation.
            - dippr_c : Parameter ``C`` of the correlation.
            - dippr_d : Parameter ``D`` of the correlation.
            - dippr_n : Exponent ``n``.
            - density : Liquid density in kg/m³.
            - molar_mass : Molar mass in kg/mol.
            - temperature : Reference temperature in kelvin.
            - phase_index : Optional array mapping species to phases.
        """

        super().__init__(temperature=temperature)
        self.dippr_a = np.asarray(dippr_a, dtype=np.float64)
        self.dippr_b = np.asarray(dippr_b, dtype=np.float64)
        self.dippr_c = np.asarray(dippr_c, dtype=np.float64)
        self.dippr_d = np.asarray(dippr_d, dtype=np.float64)
        self.dippr_n = np.asarray(dippr_n, dtype=np.float64)
        self.critical_temperature = np.asarray(
            critical_temperature, dtype=np.float64
        )
        self.temperature = float(temperature)
        self.density = density
        self.molar_mass = molar_mass
        self.phase_index = (
            None if phase_index is None else np.array(phase_index, dtype=int)
        )

    def get_surface_tension_at_temperature(
        self, temperature: Optional[float] = None
    ) -> NDArray[np.float64]:
        """
        Return surface tension at ``temperature`` using DIPPR-106.

        Arguments:
            - temperature : Optional temperature in kelvin.

        Returns:
            - Calculated surface tension in newtons per metre.

        Examples:
            ``` py title="Example"
            strat = SurfaceStrategyTemperatureMolar(0.072, 647.1)
            strat._surface_tension_at_temperature(298.0)
            ```
        """

        temp = self.temperature if temperature is None else float(temperature)
        theta = temp / self.critical_temperature
        return self.dippr_a * (1 - theta) ** self.dippr_n * (
            1
            + self.dippr_b * theta
            + self.dippr_c * theta**2
            + self.dippr_d * theta**3
        )

    def effective_surface_tension(
        self,
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """Mix surface tension using mole fractions.

        Arguments:
            - mass_concentration : Species mass concentration in kg/m³.
            - temperature : Optional temperature in kelvin.

        Returns:
            - Effective surface tension in newtons per metre.
        """
        surface_tension = self.get_surface_tension_at_temperature(temperature)
        if surface_tension.size == 1 or self.phase_index is None:
            return np.asarray(surface_tension, dtype=np.float64)

        mole_frac = get_mole_fraction_from_mass(
            mass_concentration, self.molar_mass  # type: ignore
        )
        return _weighted_average_by_phase(
            np.asarray(surface_tension, dtype=np.float64),
            mole_frac,
            self.phase_index,
        )

    def get_density(self) -> Union[float, NDArray[np.float64]]:
        return self.density

    def kelvin_radius(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
        temperature: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Return the Kelvin radius for ``temperature``.

        Arguments:
            - molar_mass : Molar mass in kg/mol.
            - mass_concentration : Species mass concentration in kg/m³.
            - temperature : Temperature in kelvin.

        Returns:
            - Kelvin radius in metres.
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
        """Calculate the Kelvin term for curvature effects.

        Arguments:
            - radius : Particle radius in metres.
            - molar_mass : Molar mass in kg/mol.
            - mass_concentration : Species mass concentration in kg/m³.
            - temperature : Temperature in kelvin.

        Returns:
            - Exponential Kelvin factor.
        """

        return get_kelvin_term(
            radius,
            self.kelvin_radius(molar_mass, mass_concentration, temperature),
        )
