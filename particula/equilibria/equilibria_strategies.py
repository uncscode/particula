"""Equilibria strategy abstractions and liquid-vapor implementation.

Provides a Strategy-pattern entry point for equilibria solvers plus
structured dataclasses for results. The initial implementation wraps the
existing liquid-vapor partitioning helpers.

References:
    Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
    Relative-humidity-dependent organic aerosol thermodynamics via an
    efficient reduced-complexity model. Atmospheric Chemistry and Physics,
    19(19), 13383-13407. https://doi.org/10.5194/acp-19-13383-2019
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence, overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

from particula.equilibria import partitioning as partitioning_module
from particula.equilibria.partitioning import (
    PhaseOutput,
    SystemOutput,
)

if TYPE_CHECKING:
    from particula_aiomfac.reservoirs import (
        SpeciesEquilibrium,
        SpeciesInventory,
    )

partitioning = partitioning_module


@dataclass
class PhaseConcentrations:
    """Concentrations in a single phase.

    Attributes:
        species_concentrations: Species concentrations (organic + aqueous)
            in the phase [µg/m³].
        water_concentration: Total aqueous concentration in the phase [µg/m³].
        total_concentration: Total concentration (organic + water) [µg/m³].
    """

    species_concentrations: NDArray[np.float64]
    water_concentration: float
    total_concentration: float


@dataclass
class EquilibriumResult:
    """Result of an equilibria calculation.

    Attributes:
        alpha_phase: Concentrations in the alpha (water-rich) phase.
        beta_phase: Concentrations in the beta (organic-rich) phase. ``None``
            when no beta phase is present.
        partition_coefficients: Species partition coefficients [-].
        water_content: Water content for (alpha, beta) phases [µg/m³].
        error: Optimization convergence error.
    """

    alpha_phase: PhaseConcentrations
    beta_phase: Optional[PhaseConcentrations]
    partition_coefficients: NDArray[np.float64]
    water_content: tuple[float, float]
    error: float


@dataclass
class MixtureEquilibriumResult(EquilibriumResult):
    """Whole-mixture bulk equilibrium with material lanes in declaration order.

    Concentrations and water exchange use µg/m³. Relative humidity is a ratio.
    The liquids record holds the phase-resolved AIOMFAC chemistry, or None
    when the condensed state is dry or the entire inventory is gaseous.
    Inspect solid_liquid for crystal and residual ion masses.
    """

    liquids: object | None
    gas_concentrations: NDArray[np.float64]
    relative_humidity: float
    water_exchanged: float
    vapor_residual: float
    solid_liquid: object | None
    phase_component_names: tuple[str, ...]


class EquilibriaStrategy(ABC):
    """Abstract base class for equilibria strategies.

    Equilibria strategies compute thermodynamic equilibrium states given
    system conditions. Concrete implementations may solve liquid-vapor,
    solid-liquid, or other phase equilibria problems.

    Examples:
        >>> import numpy as np
        >>> class CustomEquilibria(EquilibriaStrategy):
        ...     def solve(self, *args, **kwargs):
        ...         return kwargs["equilibrium_result"]
        >>> strategy = CustomEquilibria()
        >>> strategy.get_name()
        'CustomEquilibria'
    """

    @abstractmethod
    def solve(self, *args, **kwargs) -> Any:
        """Solve for equilibrium with the inputs of the configured model."""

    def get_name(self) -> str:
        """Return a stable strategy identifier."""
        return self.__class__.__name__


def _validate_closed_species_reservoir(strategy: EquilibriaStrategy) -> None:
    """Physical species inventories close the gas and condensed water budget."""
    if getattr(strategy, "mixture_water_activity", None) is not None:
        raise ValueError("species inventories use a closed gas reservoir")


class LiquidVaporPartitioningStrategy(EquilibriaStrategy):
    """Liquid-vapor partitioning equilibrium strategy.

    Wraps the partitioning helpers to compute equilibrium between gas and
    condensed phases while exposing structured results.

    Args:
        water_activity: Target water activity (0-1 range). Defaults to 0.5.

    Examples:
        >>> import numpy as np
        >>> strategy = LiquidVaporPartitioningStrategy(water_activity=0.75)
        >>> result = strategy.solve(
        ...     c_star_j_dry=np.array([1e-6, 1e-4, 1e-2]),
        ...     concentration_organic_matter=np.array([1.0, 5.0, 10.0]),
        ...     molar_mass=np.array([200.0, 200.0, 200.0]),
        ...     oxygen2carbon=np.array([0.2, 0.3, 0.5]),
        ...     density=np.array([1200.0, 1200.0, 1200.0]),
        ... )
        >>> isinstance(result.partition_coefficients, np.ndarray)
        True
    """

    def __init__(
        self, water_activity: float | None = None, *, activity_model=None
    ) -> None:
        """Initialize the strategy with the requested water activity."""
        self.mixture_water_activity = water_activity
        if water_activity is None:
            water_activity = 0.5
        if not 0 <= water_activity <= 1:
            raise ValueError(
                f"water_activity must be in [0, 1], got {water_activity}"
            )
        self.water_activity = float(water_activity)
        if activity_model is not None and not callable(
            getattr(activity_model, "solve_partitioning", None)
        ):
            raise TypeError("activity model must provide solve_partitioning")
        self.activity_model = activity_model

    @overload
    def solve(
        self,
        *,
        species_inventory: SpeciesInventory,
        temperature: float,
        **options,
    ) -> SpeciesEquilibrium: ...

    @overload
    def solve(
        self,
        *,
        inventories,
        temperature: float,
        saturation_concentrations,
        **options,
    ) -> MixtureEquilibriumResult: ...

    @overload
    def solve(
        self,
        c_star_j_dry: NDArray[np.float64],
        concentration_organic_matter: NDArray[np.float64],
        molar_mass: NDArray[np.float64],
        oxygen2carbon: NDArray[np.float64],
        density: NDArray[np.float64],
        partition_coefficient_guess: Optional[NDArray[np.float64]] = None,
    ) -> EquilibriumResult: ...

    def solve(
        self,
        c_star_j_dry: NDArray[np.float64] | None = None,
        concentration_organic_matter: NDArray[np.float64] | None = None,
        molar_mass: NDArray[np.float64] | None = None,
        oxygen2carbon: NDArray[np.float64] | None = None,
        density: NDArray[np.float64] | None = None,
        partition_coefficient_guess: Optional[NDArray[np.float64]] = None,
        **mixture_inputs,
    ) -> EquilibriumResult | SpeciesEquilibrium:
        """Solve for equilibrium state using liquid-vapor partitioning."""
        legacy = (
            c_star_j_dry,
            concentration_organic_matter,
            molar_mass,
            oxygen2carbon,
            density,
        )
        if self.activity_model is not None:
            if (
                any(value is not None for value in legacy)
                or partition_coefficient_guess is not None
            ):
                raise TypeError(
                    "configured activity model takes analytical inventories "
                    "and volatility data"
                )
            if "species_inventory" in mixture_inputs:
                _validate_closed_species_reservoir(self)
                return self.activity_model.equilibrate_species(**mixture_inputs)
            mixture_inputs.setdefault(
                "relative_humidity", self.mixture_water_activity
            )
            bulk = self.activity_model.solve_partitioning(**mixture_inputs)
            return self._mixture_result(bulk, mixture_inputs["inventories"])
        if (
            mixture_inputs
            or c_star_j_dry is None
            or concentration_organic_matter is None
            or molar_mass is None
            or oxygen2carbon is None
            or density is None
        ):
            raise TypeError(
                "BAT partitioning requires volatility, concentration, "
                "molar mass, O:C and density"
            )

        if c_star_j_dry.size == 0:
            raise ValueError("input arrays must be non-empty")

        gamma_organic_ab, mass_fraction_water_ab, q_ab = (
            partitioning.get_properties_for_liquid_vapor_partitioning(
                water_activity_desired=self.water_activity,
                molar_mass=molar_mass,
                oxygen2carbon=oxygen2carbon,
                density=density,
            )
        )

        alpha, beta, system_output, fit_result = (
            partitioning.liquid_vapor_partitioning(
                c_star_j_dry=c_star_j_dry,
                concentration_organic_matter=concentration_organic_matter,
                molar_mass=molar_mass,
                gamma_organic_ab=gamma_organic_ab,
                mass_fraction_water_ab=mass_fraction_water_ab,
                q_ab=q_ab,
                partition_coefficient_guess=partition_coefficient_guess,
            )
        )

        error_value = self._extract_error(system_output, fit_result)
        return self._convert_to_result(
            alpha=alpha,
            beta=beta,
            system=system_output,
            error_value=error_value,
        )

    def _convert_to_result(
        self,
        alpha: PhaseOutput,
        beta: Optional[PhaseOutput],
        system: SystemOutput,
        error_value: float,
    ) -> EquilibriumResult:
        if not self._is_valid_phase_output(alpha):
            raise ValueError("alpha phase output must be a length-4 sequence")

        if beta is not None and not self._is_valid_phase_output(beta):
            raise ValueError("beta phase output must be a length-4 sequence")

        if len(system) != 4:
            raise ValueError("system output must be a length-4 sequence")

        partition_coefficients = np.asarray(system[2], dtype=float)

        alpha_phase = self._to_phase(alpha)
        beta_phase = self._to_phase(beta) if beta is not None else None

        if (
            partition_coefficients.shape
            and partition_coefficients.shape
            != alpha_phase.species_concentrations.shape
        ):
            raise ValueError(
                "partition_coefficients shape must match species concentrations"
            )

        water_content = (
            float(alpha[3]),
            float(beta[3]) if beta is not None else 0.0,
        )

        return EquilibriumResult(
            alpha_phase=alpha_phase,
            beta_phase=beta_phase,
            partition_coefficients=partition_coefficients,
            water_content=water_content,
            error=float(error_value),
        )

    def _to_phase(self, phase: PhaseOutput) -> PhaseConcentrations:
        organic = np.asarray(phase[0], dtype=float)
        aqueous = np.asarray(phase[1], dtype=float)
        if organic.shape != aqueous.shape:
            raise ValueError("phase organic and aqueous arrays must align")

        species_concentrations = organic + aqueous
        water_concentration = float(phase[3])
        total_concentration = float(phase[2])

        return PhaseConcentrations(
            species_concentrations=species_concentrations,
            water_concentration=water_concentration,
            total_concentration=total_concentration,
        )

    def _mixture_result(self, bulk, inventories) -> MixtureEquilibriumResult:
        water = self.activity_model.water_index
        zero = np.zeros_like(bulk.gas_concentrations)
        names = tuple(self.activity_model.mixture.components)
        if bulk.solid_liquid is not None:
            names = bulk.solid_liquid.component_names
        alpha = (
            bulk.liquids.alpha_masses
            if bulk.liquids is not None
            else np.zeros(len(names))
        )
        beta = bulk.liquids.beta_masses if bulk.liquids is not None else None
        total = np.array(inventories, dtype=float, copy=True)
        condensed = total - bulk.gas_concentrations
        condensed[water] = alpha[water] + (
            beta[water] if beta is not None else 0.0
        )
        total[water] = bulk.gas_concentrations[water] + condensed[water]

        def phase(mass):
            return PhaseConcentrations(
                mass.copy(), float(mass[water]), float(mass.sum())
            )

        return MixtureEquilibriumResult(
            phase(alpha),
            phase(beta) if beta is not None else None,
            np.divide(condensed, total, out=zero.copy(), where=total > 0),
            (
                float(alpha[water]),
                float(beta[water]) if beta is not None else 0.0,
            ),
            max(
                bulk.vapor_residual,
                bulk.liquids.chemical_potential_error if bulk.liquids else 0.0,
            ),
            bulk.liquids,
            bulk.gas_concentrations,
            bulk.relative_humidity,
            bulk.water_exchanged,
            bulk.vapor_residual,
            bulk.solid_liquid,
            names,
        )

    @staticmethod
    def _is_valid_phase_output(phase: Sequence[object]) -> bool:
        return isinstance(phase, Sequence) and len(phase) == 4

    @staticmethod
    def _extract_error(
        system: SystemOutput, fit_result: OptimizeResult
    ) -> float:
        if len(system) >= 4 and np.isfinite(system[3]):
            return float(system[3])
        fit_value = getattr(fit_result, "fun", np.nan)
        if np.isfinite(fit_value):
            return float(fit_value)
        raise ValueError("Unable to determine finite error from system output")
