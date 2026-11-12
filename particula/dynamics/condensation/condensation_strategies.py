"""Provide condensation strategies for aerosol mass transfer.

This module defines the abstract :class:`CondensationStrategy` base class
alongside concrete implementations that compute mass transfer rates for
isothermal and staggered condensation updates. Helpers cover Knudsen-number
calculations, vapor transition corrections, and batch-aware mass adjustments
that keep updates stable when radii approach continuum limits.

The staggered implementation draws on established numerical methods:

1. **Operator Splitting**: LeVeque (2002) — Finite Volume Methods for
   Hyperbolic Problems.
2. **Symplectic Integration**: Hairer, Lubich, & Wanner (2006) — Geometric
   Numerical Integration.
3. **Mass-Conserving Condensation**: Jacobson (1997, 1999) — Analytical
   Predictor of Condensation (APC).
4. **Particle-Resolved Modeling**: Riemer et al. (2009) — PartMC-MOSAIC.

References:
    - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
      Physics: From Air Pollution to Climate Change (3rd ed.). Wiley.
    - Topping, D. & Bane, M. (2022). Introduction to Aerosol Modelling. Wiley.
      https://doi.org/10.1002/9781119625728
"""

import logging
from abc import ABC, abstractmethod
from math import fsum
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from particula.dynamics.condensation._native_step import (
    finalize_native_transfer,
    retire_evaporated_slots,
    validate_native_step_storage,
    validate_native_time_step,
)
from particula.dynamics.condensation.mass_transfer import (
    get_first_order_mass_transport_k,
    get_mass_transfer_rate,
    get_mass_transfer_rate_latent_heat,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
    condensation_population_weights,
    validate_condensation_state,
)
from particula.gas import get_molecule_mean_free_path
from particula.gas.gas_data import GasData
from particula.gas.latent_heat_strategies import (
    ConstantLatentHeat,
    LatentHeatStrategy,
)
from particula.gas.properties.thermal_conductivity import (
    get_thermal_conductivity,
)
from particula.gas.vapor_pressure_strategies import VaporPressureStrategy
from particula.particles import (
    get_knudsen_number,
    get_vapor_transition_correction,
)

# particula imports
from particula.particles.activity_strategies import ActivityStrategy
from particula.particles.particle_data import ParticleData
from particula.particles.surface_strategies import SurfaceStrategy
from particula.util.validate_inputs import validate_inputs

logger = logging.getLogger("particula")

# Minimum particle radius (0.1 nm) below which continuum mechanics breaks down
# and condensation equations are no longer physically valid
MIN_PARTICLE_RADIUS_M = 1e-10  # meters


def _pure_vapor_pressure_from_strategy(
    vapor_pressure_strategy: VaporPressureStrategy
    | Sequence[VaporPressureStrategy],
    temperature: float,
) -> NDArray[np.float64]:
    """Compute pure vapor pressure for single or per-species strategies."""
    if isinstance(vapor_pressure_strategy, Sequence):
        return np.array(
            [
                strategy.pure_vapor_pressure(temperature)
                for strategy in vapor_pressure_strategy
            ],
            dtype=np.float64,
        )
    return np.asarray(
        vapor_pressure_strategy.pure_vapor_pressure(temperature),
        dtype=np.float64,
    )


def _partial_pressure_from_strategy(
    vapor_pressure_strategy: VaporPressureStrategy
    | Sequence[VaporPressureStrategy],
    concentration: NDArray[np.float64],
    molar_mass: NDArray[np.float64],
    temperature: float,
) -> NDArray[np.float64]:
    """Compute partial pressure for single or per-species strategies."""
    if isinstance(vapor_pressure_strategy, Sequence):
        return np.array(
            [
                strategy.partial_pressure(
                    concentration=concentration[idx],
                    molar_mass=molar_mass[idx],
                    temperature=temperature,
                )
                for idx, strategy in enumerate(vapor_pressure_strategy)
            ],
            dtype=np.float64,
        )
    return np.asarray(
        vapor_pressure_strategy.partial_pressure(
            concentration=concentration,
            molar_mass=molar_mass,
            temperature=temperature,
        ),
        dtype=np.float64,
    )


def _native_kelvin_lane(
    kelvin: NDArray[np.float64],
    particle: ParticleData,
    target: int,
) -> NDArray[np.float64]:
    """Select a declared particle lane from scalar or resolved Kelvin terms."""
    if kelvin.ndim == 2:
        if kelvin.shape == (particle.n_particles, 1):
            kelvin = kelvin[:, 0]
        elif kelvin.shape == particle.masses[0].shape:
            kelvin = kelvin[:, target]
        else:
            raise ValueError("Kelvin terms must match particle lanes")
    kelvin = np.broadcast_to(kelvin, (particle.n_particles,))
    if not np.all(np.isfinite(kelvin)) or np.any(kelvin < 0):
        raise ValueError("kelvin_term must be finite and nonnegative")
    return kelvin


# mass transfer abstract class
class CondensationStrategy(ABC):
    """Abstract base class for condensation strategies.

    This class defines the interface for various condensation models
    used in atmospheric physics. Subclasses should implement specific
    condensation algorithms based on different physical models and equations.

    Attributes:
        molar_mass: The molar mass of the species [kg/mol].
        diffusion_coefficient: The diffusion coefficient [m^2/s].
        accommodation_coefficient: The mass accommodation coefficient
            (unitless).
        update_gases: Whether to update gas concentrations after
            condensation.
        skip_partitioning_indices: Optional list of indices for species
            that should not partition during condensation (default is None).

    Methods:
    - mean_free_path : Calculate the mean free path of the gas molecules.
    - knudsen_number : Compute the Knudsen number for a given particle radius.
    - first_order_mass_transport : Calculate first-order mass transport
        coefficient.
    - calculate_pressure_delta : Compute the partial pressure difference.
    - mass_transfer_rate : Abstract method for the mass transfer rate [kg/s].
    - rate : Abstract method for condensation rate per particle/bin.
    - step : Abstract method to perform one timestep of condensation.

    Concrete strategies require native containers and explicit process-owned
    activity, surface, vapor-pressure physics and species mapping. Scalar
    transport helpers may be evaluated independently of population execution.

    References:
    - Seinfeld, J. H. & Pandis, S. N. (2016). Atmospheric Chemistry and
      Physics: From Air Pollution to Climate Change (3rd ed.). Wiley.
    """

    # pylint: disable=R0913, R0917
    @validate_inputs(
        {
            "molar_mass": "positive",
            "diffusion_coefficient": "positive",
            "accommodation_coefficient": "nonnegative",
            "skip_partitioning_indices": "nonnegative",
        }
    )
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-5,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
        skip_partitioning_indices: Optional[Sequence[int]] = None,
        activity_strategy: ActivityStrategy | None = None,
        surface_strategy: SurfaceStrategy | None = None,
        vapor_pressure_strategy: VaporPressureStrategy
        | Sequence[VaporPressureStrategy]
        | None = None,
        *,
        species_map: CondensationSpeciesMap | None = None,
    ):
        """Initialize the CondensationStrategy instance.

        Args:
            molar_mass: Molar mass of the species [kg/mol].
            diffusion_coefficient: Diffusion coefficient [m^2/s].
            accommodation_coefficient: Mass accommodation coefficient
                (unitless).
            update_gases: Flag indicating whether gas concentrations should
                be updated on condensation.
            skip_partitioning_indices: Optional list of indices for species
                that should not partition during condensation (default is None).
            activity_strategy: Activity strategy used when ParticleData inputs
                are provided (required for data-only usage).
            surface_strategy: Surface strategy used when ParticleData inputs
                are provided (required for data-only usage).
            vapor_pressure_strategy: Vapor pressure strategy or per-species
                strategies in complete gas order for native inputs.
            species_map: Explicit gas-name provenance and gas-to-particle pairs
                required for tagged native data. Unmapped lanes do not transfer.
        """
        self.molar_mass = molar_mass
        self.diffusion_coefficient = diffusion_coefficient
        self.accommodation_coefficient = accommodation_coefficient
        self.update_gases = update_gases
        self.skip_partitioning_indices = skip_partitioning_indices
        self.activity_strategy = activity_strategy
        self.surface_strategy = surface_strategy
        self.vapor_pressure_strategy = vapor_pressure_strategy
        self.species_map = species_map

    def _native_parameters(
        self,
        particle: ParticleData,
        gas: GasData,
        temperature: float,
        pressure: float,
    ) -> tuple[
        NDArray[np.intp],
        NDArray[np.intp],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Admit current state and full-gas transport parameters."""
        sources, targets = validate_condensation_state(
            particle,
            gas,
            self.species_map,
            temperature,
            pressure,
            self.skip_partitioning_indices,
        )
        width = len(gas.name)
        parameters = []
        for value, label in (
            (self.molar_mass, "molar_mass"),
            (self.diffusion_coefficient, "diffusion_coefficient"),
            (self.accommodation_coefficient, "accommodation_coefficient"),
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.ndim != 0 and array.shape != (width,):
                raise ValueError(f"{label} must be scalar or full-gas ordered")
            if not np.all(np.isfinite(array)) or np.any(array <= 0):
                raise ValueError(f"{label} must be positive and finite")
            parameters.append(np.broadcast_to(array, (width,)))
        if not np.array_equal(parameters[0], gas.molar_mass):
            raise ValueError(
                "process molar_mass must match full ordered gas data"
            )
        if not isinstance(self.update_gases, (bool, np.bool_)):
            raise ValueError("update_gases must be boolean")
        return sources, targets, parameters[0], parameters[1], parameters[2]

    def _native_pressure_terms(
        self,
        particle: ParticleData,
        gas: GasData,
        temperature: float,
        radius: NDArray[np.float64],
        sources: NDArray[np.intp],
        targets: NDArray[np.intp],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate full composition and mapped pressure lanes in Pa."""
        activity, surface, vapor = self._resolve_strategies()
        if isinstance(vapor, Sequence) and len(vapor) != len(gas.name):
            raise ValueError(
                "vapor pressure strategies must follow full gas order"
            )
        pure = np.asarray(
            _pure_vapor_pressure_from_strategy(vapor, float(temperature))
            if isinstance(vapor, Sequence)
            else vapor.pure_vapor_pressure(
                np.full(len(gas.name), float(temperature))
            )
        )
        if pure.ndim != 0 and pure.shape != (len(gas.name),):
            raise ValueError(
                "vapor pressure must be scalar or full-gas ordered"
            )
        pure = np.broadcast_to(pure, (len(gas.name),))
        if not np.all(np.isfinite(pure)) or np.any(pure < 0):
            raise ValueError(
                "pure vapor pressure must be finite and nonnegative"
            )
        partial_gas = _partial_pressure_from_strategy(
            vapor, gas.concentration[0], gas.molar_mass, temperature
        )
        if partial_gas.shape != (len(gas.name),):
            raise ValueError("partial pressure must follow full gas order")
        if not np.all(np.isfinite(partial_gas)):
            raise ValueError("partial gas pressure must be finite")
        # Evaluate activity using ALL material, including particle-only lanes.
        fractions = np.asarray(
            activity.partial_pressure(
                pure_vapor_pressure=np.ones(particle.n_species),
                mass_concentration=particle.masses[0],
            ),
            dtype=np.float64,
        )
        fractions = np.broadcast_to(fractions, particle.masses[0].shape)
        if not np.all(np.isfinite(fractions)) or np.any(fractions < 0):
            raise ValueError("particle activity must be finite and nonnegative")
        delta = np.zeros_like(particle.masses[0])
        equilibrium = np.zeros_like(delta)
        if particle.n_particles == 0:
            # Gas and activity configuration were validated above. There are
            # no material radii on which to evaluate the Kelvin correction.
            return delta, equilibrium
        for source, target in zip(sources, targets, strict=True):
            kelvin = np.asarray(
                surface.kelvin_term(
                    radius=radius,
                    molar_mass=float(gas.molar_mass[source]),
                    mass_concentration=particle.masses[0],
                    temperature=temperature,
                ),
                dtype=np.float64,
            )
            kelvin = _native_kelvin_lane(kelvin, particle, int(target))
            equilibrium[:, target] = (
                pure[source] * fractions[:, target] * kelvin
            )
            delta[:, target] = partial_gas[source] - equilibrium[:, target]
        return delta, equilibrium

    def _native_mass_transfer_rate(
        self,
        particle: ParticleData,
        gas_species: GasData,
        temperature: float,
        pressure: float,
        dynamic_viscosity: float | None = None,
    ) -> NDArray[np.float64]:
        """Return native per-particle kg/s in full particle species order.

        Transport and vapor parameters follow the full gas ordering. Activity
        and surface strategies see the full particle composition. Pair-specific
        scalar transport calls avoid accidental particle/species broadcasting.
        """
        particle_data = particle
        gas = gas_species
        parameters = self._native_parameters(
            particle_data, gas, temperature, pressure
        )
        if dynamic_viscosity is not None and (
            isinstance(dynamic_viscosity, (bool, np.bool_))
            or np.ndim(dynamic_viscosity) != 0
            or not np.isfinite(dynamic_viscosity)
            or dynamic_viscosity <= 0
        ):
            raise ValueError("dynamic_viscosity must be positive and finite")
        # Resolve configuration even for no-work maps and empty populations.
        self._resolve_strategies()
        radius = np.maximum(particle_data.radii[0], MIN_PARTICLE_RADIUS_M)
        return self._native_rate_from_parameters(
            particle_data,
            gas,
            temperature,
            pressure,
            parameters,
            radius,
            dynamic_viscosity,
        )

    def _native_rate_from_parameters(
        self,
        particle_data: ParticleData,
        gas: GasData,
        temperature: float,
        pressure: float,
        parameters: tuple[
            NDArray[np.intp],
            NDArray[np.intp],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        radius: NDArray[np.float64],
        dynamic_viscosity: float | None,
    ) -> NDArray[np.float64]:
        """Evaluate admitted material lanes without rescanning a population.

        Internal batch material carriers may be untagged: population weights
        have already been admitted on the complete grid and are passed to the
        inventory finalizer separately. This helper never interprets their
        concentration as density or performs public process admission.
        """
        sources, targets, molar, diffusion, accommodation = parameters
        result = np.zeros_like(particle_data.masses[0])
        delta, equilibrium = self._native_pressure_terms(
            particle_data, gas, temperature, radius, sources, targets
        )
        heat = self._native_latent_heat_values(gas, temperature)
        for source, target in zip(sources, targets, strict=True):
            mean_free_path = get_molecule_mean_free_path(
                molar_mass=float(molar[source]),
                temperature=temperature,
                pressure=pressure,
                dynamic_viscosity=dynamic_viscosity,
            )
            transition = get_vapor_transition_correction(
                knudsen_number=mean_free_path / radius,
                mass_accommodation=float(accommodation[source]),
            )
            transport = get_first_order_mass_transport_k(
                particle_radius=radius,
                vapor_transition=transition,
                diffusion_coefficient=float(diffusion[source]),
            )
            pressure_delta = np.nan_to_num(
                delta[:, target], nan=0, posinf=0, neginf=0
            )
            if heat is None:
                result[:, target] = get_mass_transfer_rate(
                    pressure_delta, transport, temperature, float(molar[source])
                )
            else:
                result[:, target] = get_mass_transfer_rate_latent_heat(
                    pressure_delta,
                    transport,
                    temperature,
                    float(molar[source]),
                    float(heat[source]),
                    get_thermal_conductivity(temperature),
                    equilibrium[:, target],
                    float(diffusion[source]),
                )
        result[particle_data.concentration[0] == 0] = 0
        return result

    def _native_latent_heat_values(
        self, gas: GasData, temperature: float
    ) -> NDArray[np.float64] | None:
        """Validate optional latent heat in the complete gas ordering."""
        heat_strategy = getattr(self, "_latent_heat_strategy", None)
        if heat_strategy is None:
            return None
        heat = np.asarray(
            heat_strategy.latent_heat(temperature), dtype=np.float64
        )
        if heat.ndim != 0 and heat.shape != (len(gas.name),):
            raise ValueError("latent heat must be scalar or full-gas ordered")
        if not np.all(np.isfinite(heat)) or np.any(heat < 0):
            raise ValueError("latent heat must be finite and nonnegative")
        return np.broadcast_to(heat, (len(gas.name),))

    def _native_single_step(
        self,
        particle: ParticleData,
        gas: GasData,
        temperature: float,
        pressure: float,
        time_step: float,
        dynamic_viscosity: float | None,
    ) -> tuple[ParticleData, GasData]:
        """Stage one mapped isothermal/latent update, then commit by identity.

        A disabled gas update keeps the caller's reservoir fixed; the same
        finite-stage inventory limiter still bounds the particle proposal.
        Latent energy is extensive J, including resolved volume or PDF
        quadrature, and follows actual representable particle mass changes.
        """
        validate_native_time_step(time_step)
        sources, targets, _, _, _ = self._native_parameters(
            particle, gas, temperature, pressure
        )
        validate_native_step_storage(particle, gas, self.update_gases)
        weights = condensation_population_weights(particle)
        rate = np.asarray(
            self.mass_transfer_rate(
                particle, gas, temperature, pressure, dynamic_viscosity
            ),
            dtype=np.float64,
        )
        try:
            with np.errstate(over="raise", invalid="raise"):
                proposal = rate * time_step
        except FloatingPointError as error:
            raise ValueError(
                "native condensation proposal must remain finite"
            ) from error
        new_masses, new_gas = finalize_native_transfer(
            particle.masses[0],
            weights,
            gas.concentration[0],
            proposal,
            sources,
            targets,
        )
        heat = self._native_latent_heat_values(gas, temperature)
        energy = 0.0
        if heat is not None:
            energy = fsum(
                fsum(
                    (new_masses[:, target] - particle.masses[0, :, target])
                    * weights
                )
                * float(particle.volume[0])
                * float(heat[source])
                for source, target in zip(sources, targets, strict=True)
            )
            if not np.isfinite(energy):
                raise ValueError("native latent heat energy must remain finite")
        np.copyto(particle.masses[0], new_masses)
        if self.update_gases:
            np.copyto(gas.concentration[0], new_gas)
        retire_evaporated_slots(particle)
        if isinstance(self, CondensationLatentHeat):
            self.last_latent_heat_energy = energy
        return particle, gas

    def _resolve_strategies(
        self,
    ) -> tuple[
        ActivityStrategy,
        SurfaceStrategy,
        VaporPressureStrategy | Sequence[VaporPressureStrategy],
    ]:
        """Require explicit process-owned activity, surface and vapor laws."""
        if self.activity_strategy is None or self.surface_strategy is None:
            raise TypeError(
                "ParticleData requires activity_strategy and surface_strategy "
                "to be provided on the condensation strategy."
            )
        if self.vapor_pressure_strategy is None:
            raise TypeError(
                "GasData requires vapor_pressure_strategy to be provided "
                "on the condensation strategy."
            )
        return (
            self.activity_strategy,
            self.surface_strategy,
            self.vapor_pressure_strategy,
        )

    def mean_free_path(
        self,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the mean free path of the gas molecules based on the
        temperature, pressure, and dynamic viscosity of the gas.

        Args:
            temperature: The temperature of the gas [K].
            pressure: The pressure of the gas [Pa].
            dynamic_viscosity: The dynamic viscosity of the gas [Pa*s]. If
                not provided, it will be calculated based on the temperature

        Returns:
            The mean free path of the gas molecules in meters (m).

        References:
        - Mean Free Path
            [Wikipedia](https://en.wikipedia.org/wiki/Mean_free_path)

        Examples:
            ```py title="Example – Mean-free-path"
            cond = CondensationIsothermal(molar_mass=0.018)  # water vapour
            lam = cond.mean_free_path(temperature=298.15, pressure=101325)
            print(f"{lam:.2e} m")
            ```
        """
        return get_molecule_mean_free_path(
            molar_mass=self.molar_mass,
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )

    def knudsen_number(
        self,
        radius: Union[float, NDArray[np.float64]],
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """The Knudsen number for a particle.

        Calculate the Knudsen number based on the mean free path of the gas
        molecules and the radius of the particle.

        Args:
            radius: The radius of the particle [m].
            temperature: The temperature of the gas [K].
            pressure: The pressure of the gas [Pa].
            dynamic_viscosity: The dynamic viscosity of the gas [Pa*s]. If
                not provided, it will be calculated based on the temperature

        Returns:
            The Knudsen number, which is the ratio of the mean free path to
                the particle radius.

        References:
            - [Knudsen Number](https://en.wikipedia.org/wiki/Knudsen_number)

        Examples:
            ```py title="Example – Knudsen number"
            cond = CondensationIsothermal(molar_mass=0.018)
            kn = cond.knudsen_number(
                radius=1e-7, temperature=298.15, pressure=101325
            )
            ```
        """
        return get_knudsen_number(
            mean_free_path=self.mean_free_path(
                temperature=temperature,
                pressure=pressure,
                dynamic_viscosity=dynamic_viscosity,
            ),
            particle_radius=radius,
        )

    def first_order_mass_transport(
        self,
        particle_radius: Union[float, NDArray[np.float64]],
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """First-order mass transport coefficient per particle.

        Calculate the first-order mass transport coefficient, K, for a given
        particle based on the diffusion coefficient, radius, and vapor
        transition correction factor.

        Args:
            particle_radius: The radius of the particle [m].
            temperature: The temperature at which the first-order mass
                transport coefficient is to be calculated.
            pressure: The pressure of the gas phase.
            dynamic_viscosity: The dynamic viscosity of the gas [Pa*s]. If
                not provided, it will be calculated based on the temperature

        Returns:
            The first-order mass transport coefficient per particle (m^3/s).

        References:
        - Chapter 2, Equation 2.49 (excluding particle number)
        - Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling
            (D. Topping & M. Bane, Eds.). Wiley.
            [DOI](https://doi.org/10.1002/9781119625728)

        Examples:
            ```py title="Example – First-order mass-transport"
            cond = CondensationIsothermal(molar_mass=0.018)
            k = cond.first_order_mass_transport(
                particle_radius=1e-7,
                temperature=298.15,
                pressure=101325,
            )
            ```
        """
        vapor_transition = get_vapor_transition_correction(
            knudsen_number=self.knudsen_number(
                radius=particle_radius,
                temperature=temperature,
                pressure=pressure,
                dynamic_viscosity=dynamic_viscosity,
            ),
            mass_accommodation=self.accommodation_coefficient,
        )
        return get_first_order_mass_transport_k(
            particle_radius=particle_radius,
            vapor_transition=vapor_transition,
            diffusion_coefficient=self.diffusion_coefficient,
        )

    def calculate_pressure_delta(
        self,
        particle: ParticleData,
        gas_species: GasData,
        temperature: float,
        radius: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the difference in partial pressure between the gas and
        particle phases.

        Args:
            particle: The tagged single-box ParticleData for
                which the partial pressure difference is to be calculated.
            gas_species: The full ordered single-box GasData in
                contact with the particle.
            temperature: The temperature at which the partial pressure
                difference is to be calculated.
            radius: The radius of the particles.

        Returns:
            partial_pressure_delta: The difference in partial pressure
                between the gas and particle phases.

        Examples:
            ```py title="Example – Δp calculation"
            delta_p = cond.calculate_pressure_delta(
                particle=particle,
                gas_species=gas_species,
                temperature=298.15,
                radius=particle.radii[0],
            )
            ```
        """
        # Pressure-only evaluation does not consume ambient pressure.
        sources, targets, *_ = self._native_parameters(
            particle, gas_species, temperature, 1.0
        )
        radius = np.asarray(radius, dtype=np.float64)
        if radius.shape != (particle.n_particles,) or not np.all(
            np.isfinite(radius) & (radius >= 0)
        ):
            raise ValueError(
                "radius must be finite, nonnegative and per-particle"
            )
        return self._native_pressure_terms(
            particle, gas_species, temperature, radius, sources, targets
        )[0]

    @abstractmethod
    def mass_transfer_rate(
        self,
        particle: ParticleData,
        gas_species: GasData,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> NDArray[np.float64]:
        # pylint: disable=too-many-positional-arguments, too-many-arguments
        """Compute native per-particle mass transfer in full material order.

        Radii use the minimum calculation radius. Full-gas transport and vapor
        parameters are applied through the explicit species map; activity and
        surface laws see the complete particle composition. Inactive slots and
        unmapped or skipped lanes have zero rates. Nonfinite pressure
        differences
        are sanitized to zero after physical configuration validation.

        Args:
            particle: Tagged single-box ParticleData in SI units.
            gas_species: Full ordered single-box GasData.
            temperature: Positive finite scalar temperature in K.
            pressure: Positive finite scalar pressure in Pa.
            dynamic_viscosity: Optional positive finite scalar gas viscosity in
                Pa s.

        Returns:
            Per-particle rates in kg/s, shape (n_particles, n_particle_species).

        Raises:
            TypeError: If native containers or explicit process physics are
                missing.
            ValueError: If state, species mapping or physical inputs are
                invalid.
        """

    @abstractmethod
    def rate(
        self,
        particle: ParticleData,
        gas_species: GasData,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Compute each node's contribution to population mass-density change.

        Resolved counts are divided by representative volume; discrete number
        density is used directly, and radius-PDF values receive whole-grid
        quadrature weights. Species mapping and skip indices follow full gas
        order while output columns follow particle material order.

        Args:
            particle: Tagged single-box ParticleData in SI units.
            gas_species: Full ordered single-box GasData.
            temperature: Positive finite scalar temperature in K.
            pressure: Positive finite scalar pressure in Pa.

        Returns:
            Contributions in kg/m^3/s, shape (n_particles, n_particle_species).

        Raises:
            TypeError: If native containers or explicit process physics are
                missing.
            ValueError: If state, species mapping or physical inputs are
                invalid.
        """

    # pylint: disable=too-many-positional-arguments, too-many-arguments
    @abstractmethod
    def step(  # noqa: C901
        self,
        particle: ParticleData,
        gas_species: GasData,
        temperature: float,
        pressure: float,
        time_step: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Tuple[ParticleData, GasData]:
        """Advance native particle and gas state by one finite timestep.

        The step stages inventory-limited particle and gas updates and publishes
        only after successful validation and calculation. Gas updates can be
        disabled to retain an external reservoir.

        Args:
            particle: Tagged single-box ParticleData to update in place.
            gas_species: Full ordered single-box GasData.
            temperature: Positive finite scalar temperature in K.
            pressure: Positive finite scalar pressure in Pa.
            time_step: Finite nonnegative duration in seconds.
            dynamic_viscosity: Optional positive finite scalar gas viscosity in
                Pa s.

        Returns:
            The identical particle and gas containers, retaining destination
            arrays.

        Raises:
            TypeError: If native containers or explicit process physics are
                missing.
            ValueError: If state, storage, configuration or a proposal is
                invalid.
        """


# Define a condensation strategy with no latent heat of vaporization effect
class CondensationIsothermal(CondensationStrategy):
    """Condensation strategy under isothermal conditions.

    This class implements the isothermal condensation model, wherein
    temperature remains constant during mass transfer. It calculates
    condensation rates based on partial pressure differences, using
    no latent heat terms.

    Attributes:
        Inherits attributes from the base CondensationStrategy:
            molar_mass, diffusion_coefficient, etc.

    Methods:
        - mass_transfer_rate : Calculate the mass transfer rate under
            isothermal conditions.
        - rate : Get the per-particle condensation rate, accounting for
            concentration.
        - step : Advance the condensation state over a given time step.

    Examples:
        ```py title="Example Usage"
        iso_cond = CondensationIsothermal(
            molar_mass=gas_species.molar_mass,
            activity_strategy=activity,
            surface_strategy=surface,
            vapor_pressure_strategy=vapor_pressure,
            species_map=species_map,
        )
        rate_array = iso_cond.rate(particle, gas_species, 298.15, 101325)
        # particle and gas_species are native single-box containers.
        ```

    References:
        - Aerosol Modeling, Chapter 2, Equation 2.40
        - Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling
            (D. Topping & M. Bane, Eds.). Wiley.
            [DOI](https://doi.org/10.1002/9781119625728)
        - Seinfeld & Pandis, "Atmospheric Chemistry and Physics," 3rd Ed.,
          Wiley, 2016.
    """

    # pylint: disable=R0913, R0917
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-5,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
        skip_partitioning_indices: Optional[Sequence[int]] = None,
        activity_strategy: ActivityStrategy | None = None,
        surface_strategy: SurfaceStrategy | None = None,
        vapor_pressure_strategy: VaporPressureStrategy
        | Sequence[VaporPressureStrategy]
        | None = None,
        *,
        species_map: CondensationSpeciesMap | None = None,
    ):
        """Initialize the CondensationIsothermal strategy.

        Args:
            molar_mass: Molar mass of the species [kg/mol].
            diffusion_coefficient: Diffusion coefficient [m^2/s].
            accommodation_coefficient: Mass accommodation coefficient.
            update_gases: Whether to update gas concentrations on update.
            skip_partitioning_indices: Species indices that should skip
                partitioning.
            activity_strategy: Activity strategy used for ParticleData inputs.
            surface_strategy: Surface strategy used for ParticleData inputs.
            vapor_pressure_strategy: Vapor pressure strategy used for GasData
                inputs. Native parameters use the complete gas ordering.
            species_map: Explicit gas-name provenance and gas-to-particle pairs
                required for tagged native data. Unmapped lanes do not transfer.
        """
        super().__init__(
            molar_mass=molar_mass,
            diffusion_coefficient=diffusion_coefficient,
            accommodation_coefficient=accommodation_coefficient,
            update_gases=update_gases,
            skip_partitioning_indices=skip_partitioning_indices,
            activity_strategy=activity_strategy,
            surface_strategy=surface_strategy,
            vapor_pressure_strategy=vapor_pressure_strategy,
            species_map=species_map,
        )

    def mass_transfer_rate(
        self,
        particle: ParticleData,
        gas_species: GasData,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> NDArray[np.float64]:
        # pylint: disable=too-many-positional-arguments, too-many-arguments
        """Compute native per-particle mass transfer in full material order.

        Radii use the minimum calculation radius. Full-gas transport and vapor
        parameters are applied through the explicit species map; activity and
        surface laws see the complete particle composition. Inactive slots and
        unmapped or skipped lanes have zero rates. Nonfinite pressure
        differences
        are sanitized to zero after physical configuration validation.

        Args:
            particle: Tagged single-box ParticleData in SI units.
            gas_species: Full ordered single-box GasData.
            temperature: Positive finite scalar temperature in K.
            pressure: Positive finite scalar pressure in Pa.
            dynamic_viscosity: Optional positive finite scalar gas viscosity in
                Pa s.

        Returns:
            Per-particle rates in kg/s, shape (n_particles, n_particle_species).

        Raises:
            TypeError: If native containers or explicit process physics are
                missing.
            ValueError: If state, species mapping or physical inputs are
                invalid.
        """
        return self._native_mass_transfer_rate(
            particle, gas_species, temperature, pressure, dynamic_viscosity
        )

    def rate(
        self,
        particle: ParticleData,
        gas_species: GasData,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Compute each node's contribution to population mass-density change.

        Resolved counts are divided by representative volume; discrete number
        density is used directly, and radius-PDF values receive whole-grid
        quadrature weights. Species mapping and skip indices follow full gas
        order while output columns follow particle material order.

        Args:
            particle: Tagged single-box ParticleData in SI units.
            gas_species: Full ordered single-box GasData.
            temperature: Positive finite scalar temperature in K.
            pressure: Positive finite scalar pressure in Pa.

        Returns:
            Contributions in kg/m^3/s, shape (n_particles, n_particle_species).

        Raises:
            TypeError: If native containers or explicit process physics are
                missing.
            ValueError: If state, species mapping or physical inputs are
                invalid.
        """
        rates = self._native_mass_transfer_rate(
            particle, gas_species, temperature, pressure
        )
        return rates * condensation_population_weights(particle)[:, None]

    # pylint: disable=too-many-positional-arguments, too-many-arguments

    def step(  # noqa: C901
        self,
        particle: ParticleData,
        gas_species: GasData,
        temperature: float,
        pressure: float,
        time_step: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Tuple[ParticleData, GasData]:
        """Advance native particle and gas state by one finite timestep.

        The step stages inventory-limited particle and gas updates and publishes
        only after successful validation and calculation. Gas updates can be
        disabled to retain an external reservoir.

        Args:
            particle: Tagged single-box ParticleData to update in place.
            gas_species: Full ordered single-box GasData.
            temperature: Positive finite scalar temperature in K.
            pressure: Positive finite scalar pressure in Pa.
            time_step: Finite nonnegative duration in seconds.
            dynamic_viscosity: Optional positive finite scalar gas viscosity in
                Pa s.

        Returns:
            The identical particle and gas containers, retaining destination
            arrays.

        Raises:
            TypeError: If native containers or explicit process physics are
                missing.
            ValueError: If state, storage, configuration or a proposal is
                invalid.
        """
        return self._native_single_step(
            particle,
            gas_species,
            temperature,
            pressure,
            time_step,
            dynamic_viscosity,
        )


class CondensationIsothermalStaggered(CondensationStrategy):
    """Staggered isothermal condensation with two-pass Gauss-Seidel updates.

    This strategy splits each timestep into two passes with inventory-limited
    updates for explicitly tagged particle populations. Each pass
    executes Gauss-Seidel sweeps over batches of particles and optionally
    updates the shared gas field after every batch to reduce lag.

    theta_mode controls how the first-pass fraction (theta) is selected:
        - ``"half"`` uses a deterministic half-step (theta = 0.5) for all
          particles.
        - ``"random"`` samples theta ~ U[0, 1] with the configured RNG.
        - ``"batch"`` fixes theta = 1.0 and staggers updates through batches.

    Attributes:
        theta_mode: Staggered stepping mode, one of ``("half", "random",
            "batch")``.
        num_batches: Requested Gauss-Seidel batch count clipped to the particle
            total to avoid empty batches.
        shuffle_each_step: When True, particle order is reshuffled each step;
            set False for deterministic sequencing.
        random_state: Optional seed, ``numpy.random.Generator``, or
            ``RandomState`` reused for shuffling and theta draws to keep the
            stochastic steps reproducible.

    Example:
        >>> from particula.dynamics.condensation import (
        ...     CondensationIsothermalStaggered,
        ... )
        >>> strategy = CondensationIsothermalStaggered(
        ...     molar_mass=gas_species.molar_mass,
        ...     activity_strategy=activity,
        ...     surface_strategy=surface,
        ...     vapor_pressure_strategy=vapor_pressure,
        ...     species_map=species_map,
        ...     theta_mode="random",
        ...     num_batches=8,
        ...     shuffle_each_step=True,
        ...     random_state=1234,
        ... )
        >>> particle, gas = strategy.step(
        ...     particle, gas_species, 298.0, 101325.0, 0.1
        ... )

    See Also:
        CondensationIsothermal: Simultaneous (non-staggered) condensation.
        CondensationIsothermalStaggeredBuilder: Fluent builder for this
            strategy.
    """

    VALID_THETA_MODES = ("half", "random", "batch")

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        theta_mode: str = "half",
        num_batches: int = 1,
        shuffle_each_step: bool = True,
        random_state: Optional[
            Union[int, np.random.Generator, np.random.RandomState]
        ] = None,
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-5,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
        skip_partitioning_indices: Optional[Sequence[int]] = None,
        activity_strategy: ActivityStrategy | None = None,
        surface_strategy: SurfaceStrategy | None = None,
        vapor_pressure_strategy: VaporPressureStrategy
        | Sequence[VaporPressureStrategy]
        | None = None,
        *,
        species_map: CondensationSpeciesMap | None = None,
    ):
        """Initialize the staggered condensation strategy.

        This constructor configures theta mode, batching, and RNG options while
        reusing diffusion and accommodation settings from the base strategy. The
        requested batch count is clipped to the particle inventory to prevent
        empty Gauss-Seidel sweeps.

        Args:
            molar_mass: Molar mass of the condensing species [kg/mol].
            theta_mode: Staggered stepping mode; one of ``("half", "random",
                "batch")``.
            num_batches: Requested Gauss-Seidel batch count (>= 1).
            shuffle_each_step: When True, shuffle particle order each step;
                set False for deterministic sequencing.
            random_state: Optional seed, ``numpy.random.Generator``, or
                ``RandomState`` reused for shuffling and theta draws.
            diffusion_coefficient: Diffusion coefficient [m^2/s].
            accommodation_coefficient: Mass accommodation coefficient.
            update_gases: Whether to update gas concentrations.
            skip_partitioning_indices: Species indices to skip partitioning.
            activity_strategy: Activity strategy used for ParticleData inputs.
            surface_strategy: Surface strategy used for ParticleData inputs.
            vapor_pressure_strategy: Vapor pressure strategy used for GasData
                inputs. Native parameters use the complete gas ordering.
            species_map: Explicit gas-name provenance and gas-to-particle pairs
                required for tagged native data. Unmapped lanes do not transfer.

        Raises:
            ValueError: If ``theta_mode`` is unsupported or ``num_batches`` is
                less than 1.
        """
        super().__init__(
            molar_mass=molar_mass,
            diffusion_coefficient=diffusion_coefficient,
            accommodation_coefficient=accommodation_coefficient,
            update_gases=update_gases,
            skip_partitioning_indices=skip_partitioning_indices,
            activity_strategy=activity_strategy,
            surface_strategy=surface_strategy,
            vapor_pressure_strategy=vapor_pressure_strategy,
            species_map=species_map,
        )

        if theta_mode not in self.VALID_THETA_MODES:
            raise ValueError(
                f"theta_mode must be one of {self.VALID_THETA_MODES}, got "
                f"'{theta_mode}'"
            )
        if num_batches < 1:
            raise ValueError("num_batches must be >= 1.")

        self.theta_mode = theta_mode
        self.num_batches = num_batches
        self.shuffle_each_step = shuffle_each_step
        self.random_state = random_state

    def _get_theta_values(self, n_particles: int) -> NDArray[np.float64]:
        """Generate first-pass theta fractions for each particle.

        Theta splits the timestep into theta and (1 - theta) passes, controlling
        how much work occurs during the first Gauss-Seidel sweep. The
        configured RNG determines theta draws when ``theta_mode`` is
        ``"random"``.

        Args:
            n_particles: Number of particles requiring theta fractions.

        Returns:
            Array of shape ``(n_particles,)``:
                - ``"half"`` returns 0.5 for every particle.
                - ``"random"`` draws from ``U[0, 1]`` with the configured RNG
                  or seed so the samples remain reproducible.
                - ``"batch"`` returns 1.0 because batching provides the
                  staggering instead of theta.

        Raises:
            ValueError: When ``theta_mode`` is unsupported.
        """
        if self.theta_mode == "half":
            return np.full(n_particles, 0.5, dtype=np.float64)

        if self.theta_mode == "random":
            if isinstance(self.random_state, np.random.Generator):
                rng = self.random_state
                return rng.uniform(0.0, 1.0, n_particles)
            if isinstance(self.random_state, np.random.RandomState):
                return self.random_state.random(n_particles).astype(
                    np.float64, copy=False
                )

            rng = np.random.default_rng(self.random_state)
            return rng.uniform(0.0, 1.0, n_particles)

        if self.theta_mode == "batch":
            return np.ones(n_particles, dtype=np.float64)

        raise ValueError(f"Invalid theta_mode: {self.theta_mode}")

    def _validate_num_batches(self, num_batches: int, n_particles: int) -> int:
        """Ensure batch count respects available particles.

        Args:
            num_batches: Requested number of Gauss-Seidel batches.
            n_particles: Number of particles available for batching.

        Returns:
            Adjusted batch count clipped to ``n_particles`` when the request
            exceeds the available particles; logs the clipping action.

        Raises:
            ValueError: When ``num_batches`` is less than 1.

        Notes:
            The caller skips this helper when ``n_particles`` is zero, so no
            logging occurs for the empty case.
        """
        if num_batches < 1:
            raise ValueError("num_batches must be >= 1.")

        if num_batches > n_particles and n_particles > 0:
            logger.info(
                "Clipping num_batches from %s to %s to avoid empty batches",
                num_batches,
                n_particles,
            )
            # Clipping avoids empty batches while preserving all particles.
            return n_particles

        return num_batches

    def _make_batches(self, n_particles: int) -> list[NDArray[np.intp]]:
        """Partition particle indices into Gauss-Seidel batches.

        Args:
            n_particles: Number of particles in the simulation.

        Returns:
            List of arrays, one per batch, or an empty list when ``n_particles``
            is zero.

        Notes:
            Clips and logs when ``num_batches`` exceeds ``n_particles`` to avoid
            empty batches while preserving all particles. Shuffling uses the
            configured seed, ``Generator``, or ``RandomState`` when
            ``shuffle_each_step`` is True; order is preserved otherwise.
        """
        if n_particles == 0:
            return []

        effective_batches = self._validate_num_batches(
            self.num_batches, n_particles
        )

        indices = np.arange(n_particles, dtype=np.intp)

        if self.shuffle_each_step:
            if isinstance(self.random_state, np.random.Generator):
                self.random_state.shuffle(indices)
            elif isinstance(self.random_state, np.random.RandomState):
                self.random_state.shuffle(indices)
            else:
                rng = np.random.default_rng(self.random_state)
                rng.shuffle(indices)

        return list(np.array_split(indices, effective_batches))

    def _validate_native_staggered_controls(self) -> None:
        """Revalidate mutable controls before drawing randomness or writing."""
        if self.theta_mode not in self.VALID_THETA_MODES:
            raise ValueError("invalid native theta_mode")
        if (
            isinstance(self.num_batches, (bool, np.bool_))
            or not isinstance(self.num_batches, (int, np.integer))
            or self.num_batches < 1
        ):
            raise ValueError("num_batches must be a positive integer")
        if not isinstance(self.shuffle_each_step, (bool, np.bool_)):
            raise ValueError("shuffle_each_step must be boolean")
        random = self.random_state
        if random is None or isinstance(
            random, (np.random.Generator, np.random.RandomState)
        ):
            return
        if (
            isinstance(random, (bool, np.bool_))
            or not isinstance(random, (int, np.integer))
            or random < 0
        ):
            raise ValueError(
                "random_state must be a generator or nonnegative seed"
            )

    def _native_staggered_step(
        self,
        particle: ParticleData,
        gas: GasData,
        temperature: float,
        pressure: float,
        time_step: float,
        dynamic_viscosity: float | None,
    ) -> tuple[ParticleData, GasData]:
        """Apply two conservative Gauss-Seidel sweeps to private working state.

        Preserve the existing frozen particle composition, radii and transport
        over both sweeps. Gas pressure reads the latest batch inventory, while
        evaporation is bounded by actual remaining staged particle mass.
        Local theta durations are applied per node, including mixed durations
        in a vectorized batch. PDF weights come from the entire original grid.
        A disabled gas update retains fixed reservoir pressure between batches.
        """
        validate_native_time_step(time_step)
        self._validate_native_staggered_controls()
        parameters = self._native_parameters(
            particle, gas, temperature, pressure
        )
        sources, targets = parameters[:2]
        validate_native_step_storage(particle, gas, self.update_gases)
        # Resolve all physical/model configuration before touching caller RNG.
        self._native_mass_transfer_rate(
            particle, gas, temperature, pressure, dynamic_viscosity
        )
        weights = condensation_population_weights(particle)
        n_particles = particle.n_particles
        if time_step == 0 or n_particles == 0 or sources.size == 0:
            return particle, gas
        radius = np.maximum(particle.radii[0], MIN_PARTICLE_RADIUS_M)
        theta = self._get_theta_values(n_particles)
        batches = self._make_batches(n_particles)
        working_masses = particle.masses[0].copy()
        working_gas = gas.copy()
        for duration in (theta * time_step, (1 - theta) * time_step):
            for batch in batches:
                indices = batch[(duration[batch] > 0) & (weights[batch] > 0)]
                if indices.size == 0:
                    continue
                # A material-only slice, not a separately interpreted PDF.
                material = ParticleData(
                    masses=particle.masses[:, indices, :],
                    concentration=particle.concentration[:, indices],
                    charge=particle.charge[:, indices],
                    density=particle.density,
                    volume=particle.volume,
                )
                rate = self._native_rate_from_parameters(
                    material,
                    working_gas,
                    temperature,
                    pressure,
                    parameters,
                    radius[indices],
                    dynamic_viscosity,
                )
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        proposal = rate * duration[indices, None]
                except FloatingPointError as error:
                    raise ValueError(
                        "native staggered proposal overflow"
                    ) from error
                updated, remaining = finalize_native_transfer(
                    working_masses[indices],
                    weights[indices],
                    working_gas.concentration[0],
                    proposal,
                    sources,
                    targets,
                )
                working_masses[indices] = updated
                if particle.distribution_type == "particle_resolved":
                    weights[indices[np.all(updated == 0, axis=1)]] = 0
                if self.update_gases:
                    working_gas.concentration[0] = remaining
        np.copyto(particle.masses[0], working_masses)
        if self.update_gases:
            np.copyto(gas.concentration, working_gas.concentration)
        retire_evaporated_slots(particle)
        return particle, gas

    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def mass_transfer_rate(
        self,
        particle: ParticleData,
        gas_species: GasData,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """Compute native per-particle mass transfer in full material order.

        Radii use the minimum calculation radius. Full-gas transport and vapor
        parameters are applied through the explicit species map; activity and
        surface laws see the complete particle composition. Inactive slots and
        unmapped or skipped lanes have zero rates. Nonfinite pressure
        differences
        are sanitized to zero after physical configuration validation.

        Args:
            particle: Tagged single-box ParticleData in SI units.
            gas_species: Full ordered single-box GasData.
            temperature: Positive finite scalar temperature in K.
            pressure: Positive finite scalar pressure in Pa.
            dynamic_viscosity: Optional positive finite scalar gas viscosity in
                Pa s.

        Returns:
            Per-particle rates in kg/s, shape (n_particles, n_particle_species).

        Raises:
            TypeError: If native containers or explicit process physics are
                missing.
            ValueError: If state, species mapping or physical inputs are
                invalid.
        """
        return self._native_mass_transfer_rate(
            particle, gas_species, temperature, pressure, dynamic_viscosity
        )

    def rate(
        self,
        particle: ParticleData,
        gas_species: GasData,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Compute each node's contribution to population mass-density change.

        Resolved counts are divided by representative volume; discrete number
        density is used directly, and radius-PDF values receive whole-grid
        quadrature weights. Species mapping and skip indices follow full gas
        order while output columns follow particle material order.

        Args:
            particle: Tagged single-box ParticleData in SI units.
            gas_species: Full ordered single-box GasData.
            temperature: Positive finite scalar temperature in K.
            pressure: Positive finite scalar pressure in Pa.

        Returns:
            Contributions in kg/m^3/s, shape (n_particles, n_particle_species).

        Raises:
            TypeError: If native containers or explicit process physics are
                missing.
            ValueError: If state, species mapping or physical inputs are
                invalid.
        """
        rates = self._native_mass_transfer_rate(
            particle, gas_species, temperature, pressure
        )
        return rates * condensation_population_weights(particle)[:, None]

    # pylint: disable=too-many-positional-arguments, too-many-arguments

    def step(  # noqa: C901
        self,
        particle: ParticleData,
        gas_species: GasData,
        temperature: float,
        pressure: float,
        time_step: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Tuple[ParticleData, GasData]:
        """Advance native particle and gas state by one finite timestep.

        Two theta sweeps evaluate vectorized batches with sequential gas
        coupling. Material radii/composition are frozen for rate evaluation;
        remaining staged mass and current gas limit each proposal. Array writes
        publish after staging; consumed random draws are not rolled back.

        Args:
            particle: Tagged single-box ParticleData to update in place.
            gas_species: Full ordered single-box GasData.
            temperature: Positive finite scalar temperature in K.
            pressure: Positive finite scalar pressure in Pa.
            time_step: Finite nonnegative duration in seconds.
            dynamic_viscosity: Optional positive finite scalar gas viscosity in
                Pa s.

        Returns:
            The identical particle and gas containers, retaining destination
            arrays.

        Raises:
            TypeError: If native containers or explicit process physics are
                missing.
            ValueError: If state, storage, configuration or a proposal is
                invalid.
        """
        return self._native_staggered_step(
            particle,
            gas_species,
            temperature,
            pressure,
            time_step,
            dynamic_viscosity,
        )


class CondensationLatentHeat(CondensationStrategy):
    """Condensation strategy with optional latent-heat corrections.

    This strategy mirrors the isothermal workflow while optionally applying a
    latent-heat correction to the mass-transfer rate. When a latent heat
    strategy is supplied, the correction uses thermal resistance and the vapor
    pressure at the particle surface. Without a strategy, the behavior matches
    the isothermal implementation. Negative or array-like latent heat inputs
    log warnings and fall back to the isothermal path.

    Attributes:
        latent_heat_strategy_input: Strategy input provided at initialization.
        latent_heat_input: Raw latent heat value provided at initialization.
        last_latent_heat_energy: Total latent heat energy released in the most
            recent step [J]. Positive values indicate condensation and
            negative values indicate evaporation. Overwritten each call.
    """

    # pylint: disable=R0913, R0917
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-5,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
        skip_partitioning_indices: Optional[Sequence[int]] = None,
        activity_strategy: ActivityStrategy | None = None,
        surface_strategy: SurfaceStrategy | None = None,
        vapor_pressure_strategy: VaporPressureStrategy
        | Sequence[VaporPressureStrategy]
        | None = None,
        *,
        species_map: CondensationSpeciesMap | None = None,
        latent_heat_strategy: LatentHeatStrategy | None = None,
        latent_heat: float | NDArray[np.float64] = 0.0,
    ):
        """Initialize the CondensationLatentHeat strategy.

        Args:
            molar_mass: Molar mass of the species [kg/mol].
            diffusion_coefficient: Diffusion coefficient [m^2/s].
            accommodation_coefficient: Mass accommodation coefficient.
            update_gases: Whether to update gas concentrations on update.
            skip_partitioning_indices: Species indices that should skip
                partitioning.
            activity_strategy: Activity strategy used for ParticleData inputs.
            surface_strategy: Surface strategy used for ParticleData inputs.
            vapor_pressure_strategy: Vapor pressure strategy used for GasData
                inputs. Native parameters use the complete gas ordering.
            species_map: Explicit gas-name provenance and gas-to-particle pairs
                required for tagged native data. Unmapped lanes do not transfer.
            latent_heat_strategy: Optional latent heat strategy to use.
            latent_heat: Scalar latent heat fallback [J/kg].
        """
        super().__init__(
            molar_mass=molar_mass,
            diffusion_coefficient=diffusion_coefficient,
            accommodation_coefficient=accommodation_coefficient,
            update_gases=update_gases,
            skip_partitioning_indices=skip_partitioning_indices,
            activity_strategy=activity_strategy,
            surface_strategy=surface_strategy,
            vapor_pressure_strategy=vapor_pressure_strategy,
            species_map=species_map,
        )
        self.latent_heat_strategy_input = latent_heat_strategy
        self.latent_heat_input = latent_heat
        self._latent_heat_strategy = self._resolve_latent_heat_strategy(
            latent_heat_strategy=latent_heat_strategy,
            latent_heat=latent_heat,
        )
        self.last_latent_heat_energy = 0.0

    def _resolve_latent_heat_strategy(
        self,
        latent_heat_strategy: LatentHeatStrategy | None,
        latent_heat: float | NDArray[np.float64],
    ) -> LatentHeatStrategy | None:
        """Resolve the latent heat strategy from inputs.

        Prefers an explicit strategy, otherwise converts scalar latent heat
        values into a constant strategy. Array-like or negative latent heat
        values log a warning and fall back to isothermal behavior.

        Args:
            latent_heat_strategy: Optional strategy to use directly.
            latent_heat: Scalar or array-like latent heat input [J/kg].

        Returns:
            Resolved latent heat strategy, or None for isothermal fallback.
        """
        if latent_heat_strategy is not None:
            return latent_heat_strategy

        latent_heat_array = np.asarray(latent_heat, dtype=np.float64)
        if latent_heat_array.shape == ():
            latent_heat_value = float(latent_heat_array)
            if latent_heat_value > 0:
                return ConstantLatentHeat(latent_heat_value)
            if latent_heat_value < 0:
                logger.warning(
                    "Negative latent_heat provided; falling back to "
                    "isothermal behavior."
                )
            return None

        logger.warning(
            "Array-like latent_heat provided; use a LatentHeatStrategy "
            "for per-species values."
        )
        return None

    def mass_transfer_rate(
        self,
        particle: ParticleData,
        gas_species: GasData,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """Compute native per-particle mass transfer in full material order.

        Radii use the minimum calculation radius. Full-gas transport and vapor
        parameters are applied through the explicit species map; activity and
        surface laws see the complete particle composition. Inactive slots and
        unmapped or skipped lanes have zero rates. Nonfinite pressure
        differences
        are sanitized to zero after physical configuration validation.

        Args:
            particle: Tagged single-box ParticleData in SI units.
            gas_species: Full ordered single-box GasData.
            temperature: Positive finite scalar temperature in K.
            pressure: Positive finite scalar pressure in Pa.
            dynamic_viscosity: Optional positive finite scalar gas viscosity in
                Pa s.

        Returns:
            Per-particle rates in kg/s, shape (n_particles, n_particle_species).

        Raises:
            TypeError: If native containers or explicit process physics are
                missing.
            ValueError: If state, species mapping or physical inputs are
                invalid.
        """
        return self._native_mass_transfer_rate(
            particle, gas_species, temperature, pressure, dynamic_viscosity
        )

    def rate(
        self,
        particle: ParticleData,
        gas_species: GasData,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Compute each node's contribution to population mass-density change.

        Resolved counts are divided by representative volume; discrete number
        density is used directly, and radius-PDF values receive whole-grid
        quadrature weights. Species mapping and skip indices follow full gas
        order while output columns follow particle material order.

        Args:
            particle: Tagged single-box ParticleData in SI units.
            gas_species: Full ordered single-box GasData.
            temperature: Positive finite scalar temperature in K.
            pressure: Positive finite scalar pressure in Pa.

        Returns:
            Contributions in kg/m^3/s, shape (n_particles, n_particle_species).

        Raises:
            TypeError: If native containers or explicit process physics are
                missing.
            ValueError: If state, species mapping or physical inputs are
                invalid.
        """
        rates = self._native_mass_transfer_rate(
            particle, gas_species, temperature, pressure
        )
        return rates * condensation_population_weights(particle)[:, None]

    # pylint: disable=too-many-positional-arguments, too-many-arguments

    def step(  # noqa: C901
        self,
        particle: ParticleData,
        gas_species: GasData,
        temperature: float,
        pressure: float,
        time_step: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Tuple[ParticleData, GasData]:
        """Advance native particle and gas state by one finite timestep.

        The step stages inventory-limited particle and gas updates and publishes
        only after successful validation and calculation. Gas updates can be
        disabled to retain an external reservoir.
        Latent energy is extensive J from actual applied mass changes, positive
        for condensation and negative for evaporation; each call overwrites it.

        Args:
            particle: Tagged single-box ParticleData to update in place.
            gas_species: Full ordered single-box GasData.
            temperature: Positive finite scalar temperature in K.
            pressure: Positive finite scalar pressure in Pa.
            time_step: Finite nonnegative duration in seconds.
            dynamic_viscosity: Optional positive finite scalar gas viscosity in
                Pa s.

        Returns:
            The identical particle and gas containers, retaining destination
            arrays.

        Raises:
            TypeError: If native containers or explicit process physics are
                missing.
            ValueError: If state, storage, configuration or a proposal is
                invalid.
        """
        return self._native_single_step(
            particle,
            gas_species,
            temperature,
            pressure,
            time_step,
            dynamic_viscosity,
        )
