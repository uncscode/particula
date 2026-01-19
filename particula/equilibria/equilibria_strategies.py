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
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

from particula.equilibria import partitioning as partitioning_module
from particula.equilibria.partitioning import (
    PhaseOutput,
    SystemOutput,
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
    def solve(
        self,
        c_star_j_dry: NDArray[np.float64],
        concentration_organic_matter: NDArray[np.float64],
        molar_mass: NDArray[np.float64],
        oxygen2carbon: NDArray[np.float64],
        density: NDArray[np.float64],
        partition_coefficient_guess: Optional[NDArray[np.float64]] = None,
    ) -> EquilibriumResult:
        """Solve for equilibrium state.

        Args:
            c_star_j_dry: Dry saturation concentrations [µg/m³].
            concentration_organic_matter: Organic mass concentrations [µg/m³].
            molar_mass: Species molar masses [g/mol].
            oxygen2carbon: Oxygen-to-carbon ratios [-].
            density: Species densities [kg/m³].
            partition_coefficient_guess: Optional initial guesses for
                partition coefficients [-].

        Returns:
            Structured equilibrium result.
        """

    def get_name(self) -> str:
        """Return a stable strategy identifier."""
        return self.__class__.__name__


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

    def __init__(self, water_activity: float = 0.5) -> None:
        """Initialize the strategy with the requested water activity."""
        if not 0 <= water_activity <= 1:
            raise ValueError(
                f"water_activity must be in [0, 1], got {water_activity}"
            )
        self.water_activity = float(water_activity)

    def solve(
        self,
        c_star_j_dry: NDArray[np.float64],
        concentration_organic_matter: NDArray[np.float64],
        molar_mass: NDArray[np.float64],
        oxygen2carbon: NDArray[np.float64],
        density: NDArray[np.float64],
        partition_coefficient_guess: Optional[NDArray[np.float64]] = None,
    ) -> EquilibriumResult:
        """Solve for equilibrium state using liquid-vapor partitioning."""
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
