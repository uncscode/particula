"""Wall loss strategy abstractions and implementations.

Defines abstract and concrete strategies for modeling particle wall
loss processes in different chamber geometries. Strategies operate on
:class:`~particula.particles.particle_data.ParticleData`
objects with explicit distribution metadata. Implementations are
provided for spherical and rectangular chambers.

Strategies require explicitly tagged single-box ``ParticleData``.
Native PMF/PDF values are densities at unit volume; resolved values are counts
and rates use count/volume. Native steps preserve caller-owned arrays and stage
updates before publication. Facade inputs are no longer supported.

The wall loss rate is modeled as a first-order size dependent loss
process

.. math::

    L = -k (Dp) c,

where :math:`L` is the wall loss rate [#/m^3 s],
:math:`k` is the wall loss coefficient [1/s], and :math:`c` is the
particle number concentration [#/m^3].

References:
    Crump, J. G., & Seinfeld, J. H. (1981). Turbulent deposition and
    gravitational sedimentation of an aerosol in a vessel of arbitrary
    shape. Journal of Aerosol Science, 12(5), 405–415.
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from particula.dynamics.properties.wall_loss_coefficient import (
    get_rectangle_wall_loss_coefficient_via_system_state,
    get_spherical_wall_loss_coefficient_via_system_state,
)
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.particles.particle_data import ParticleData
from particula.particles.properties.coulomb_enhancement import (
    get_coulomb_enhancement_ratio,
)
from particula.util.constants import ELEMENTARY_CHARGE_VALUE
from particula.util.validate_inputs import validate_inputs

from . import _native


def get_particle_resolved_wall_loss_step(
    particle_radius: NDArray[np.float64],
    particle_density: NDArray[np.float64],
    concentration: NDArray[np.float64],
    loss_coefficient_func: Callable[
        [NDArray[np.float64], NDArray[np.float64]],
        NDArray[np.float64],
    ],
    time_step: float,
    random_generator: np.random.Generator,
) -> NDArray[np.float64]:
    """Perform particle-resolved wall loss step with stochastic removal.

    For particle-resolved simulations, each computational particle survives
    with probability ``exp(-k * dt)``, where ``k`` is the size-dependent wall
    loss coefficient. Binomial draws determine which particles are lost to the
    walls.

    Args:
        particle_radius: Array of particle radii [m].
        particle_density: Array of particle densities [kg/m^3].
        concentration: Array of particle concentrations [#/m^3].
        loss_coefficient_func: Callable returning wall loss coefficient for
            provided particle radius and density. Temperature, pressure, and
            geometry should be bound via closure.
        time_step: Time step [s].
        random_generator: Random number generator for stochastic draws.

    Returns:
        Array of survival indicators (1.0 for survived, 0.0 for lost).

    Examples:
        >>> import numpy as np
        >>> from particula.dynamics.wall_loss.wall_loss_strategies import (
        ...     get_particle_resolved_wall_loss_step,
        ... )
        >>> radius = np.array([1e-7, 2e-7, 3e-7])
        >>> density = np.array([1000.0, 1000.0, 1000.0])
        >>> concentration = np.array([1e6, 1e6, 1e6])
        >>> def coeff_func(r, d):
        ...     return 1e-4 * np.ones_like(r)
        >>> rng = np.random.default_rng(42)
        >>> survived = get_particle_resolved_wall_loss_step(
        ...     particle_radius=radius,
        ...     particle_density=density,
        ...     concentration=concentration,
        ...     loss_coefficient_func=coeff_func,
        ...     time_step=1.0,
        ...     random_generator=rng,
        ... )
        >>> print(survived)
        [1. 1. 1.]
    """
    # Only calculate for active particles (radius > 0)
    active_particles = particle_radius > 0

    if not np.any(active_particles):
        # No active particles, all lost
        return np.zeros_like(concentration, dtype=np.float64)

    # Calculate loss coefficient only for active particles
    active_radius = particle_radius[active_particles]
    active_density = particle_density[active_particles]
    active_coefficient = loss_coefficient_func(active_radius, active_density)

    # Calculate survival probability for active particles
    # survival_probability = exp(-k * dt)
    survival_probability_active = np.exp(-active_coefficient * time_step)
    # Clamp to [0, 1] for numerical safety
    survival_probability_active = np.clip(survival_probability_active, 0.0, 1.0)

    # Draw survival for each active particle using binomial(1, p)
    survived = np.zeros_like(concentration, dtype=np.float64)
    survived[active_particles] = random_generator.binomial(
        n=1,
        p=survival_probability_active,
    )

    return survived


class WallLossStrategy(ABC):
    """Abstract base class for wall loss strategies.

    Wall loss strategies compute wall loss coefficients and associated
    rates for different distribution types.

    Attributes:
        wall_eddy_diffusivity: Wall eddy diffusivity [1/s].
        distribution_type: Distribution type ("discrete",
            "continuous_pdf", or "particle_resolved").
        random_generator: Random number generator for stochastic
            particle-resolved simulations.
    """

    wall_eddy_diffusivity: float
    distribution_type: str
    random_generator: np.random.Generator

    @validate_inputs({"wall_eddy_diffusivity": "positive"})
    def __init__(
        self, wall_eddy_diffusivity: float, distribution_type: str
    ) -> None:
        """Initialize the wall loss strategy.

        Args:
            wall_eddy_diffusivity: Wall eddy diffusivity [1/s].
            distribution_type: Distribution type ("discrete",
                "continuous_pdf", or "particle_resolved").

        Raises:
            ValueError: If ``distribution_type`` is not supported.
        """
        if distribution_type not in [
            "discrete",
            "continuous_pdf",
            "particle_resolved",
        ]:
            raise ValueError(
                "Invalid distribution type. Must be one of 'discrete', "
                + "'continuous_pdf', or 'particle_resolved'."
            )

        self.wall_eddy_diffusivity = wall_eddy_diffusivity
        self.distribution_type = distribution_type
        self.random_generator = np.random.default_rng()

    @abstractmethod
    def loss_coefficient(
        self,
        particle: ParticleData,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Return the wall loss coefficient for the given state.

        Args:
            particle: Native particle data to evaluate.
            temperature: Gas temperature [K].
            pressure: Gas pressure [Pa].

        Returns:
            Wall loss coefficient [1/s].
        """

    @abstractmethod
    def loss_coefficient_for_particles(
        self,
        particle_radius: NDArray[np.float64],
        particle_density: NDArray[np.float64],
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Return the wall loss coefficient for given particle properties.

        This method is used for particle-resolved simulations where we
        need to calculate coefficients for a subset of active particles.

        Args:
            particle_radius: Particle radii [m].
            particle_density: Particle densities [kg/m^3].
            temperature: Gas temperature [K].
            pressure: Gas pressure [Pa].

        Returns:
            Wall loss coefficient [1/s] for each particle.
        """

    def loss_rate(
        self,
        particle: ParticleData,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Return the wall loss rate for the given state.

        The loss rate is computed as ``-k(Dp) * c`` where ``k(Dp)`` is the
        size-dependent wall loss coefficient and ``c`` is the particle number
        concentration.

        Args:
            particle: Native particle data to evaluate.
            temperature: Gas temperature [K].
            pressure: Gas pressure [Pa].

        Returns:
            Wall loss rate [#/m^3 s] (negative values indicate loss).
        """
        return _native.rate(self, particle, temperature, pressure)

    def rate(
        self,
        particle: ParticleData,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Return the wall loss rate as an array.

        Args:
            particle: Native particle data to evaluate.
            temperature: Gas temperature [K].
            pressure: Gas pressure [Pa].

        Returns:
            Array of wall loss rates [#/m^3 s].
        """
        return np.asarray(
            self.loss_rate(
                particle=particle,
                temperature=temperature,
                pressure=pressure,
            )
        )

    def step(
        self,
        particle: ParticleData,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> ParticleData:
        """Advance the native particle data by one wall loss step.

        For ``"discrete"`` and ``"continuous_pdf"`` distributions, this
        applies a deterministic first-order loss to the number
        concentration in each bin. For ``"particle_resolved"``, it
        applies a stochastic removal process using binomial random draws
        based on survival probability.

        Args:
            particle: Native particle data to update.
            temperature: Gas temperature [K].
            pressure: Gas pressure [Pa].
            time_step: Time step [s].

        Returns:
            Updated native particle data (same instance).

        Raises:
            ValueError: If ``distribution_type`` is not supported.
        """
        return _native.step(self, particle, temperature, pressure, time_step)


class SphericalWallLossStrategy(WallLossStrategy):
    """Wall loss strategy for spherical chambers.

    Calculates particle wall deposition in spherical chamber geometry
    using turbulent diffusion and gravitational settling.

    Attributes:
        wall_eddy_diffusivity: Wall eddy diffusivity [1/s].
        chamber_radius: Radius of the spherical chamber [m].
        distribution_type: Distribution type ("discrete",
            "continuous_pdf", or "particle_resolved").

    Examples:
        >>> from particula.dynamics.wall_loss.wall_loss_strategies import (
        ...     SphericalWallLossStrategy,
        ... )
        >>> strategy = SphericalWallLossStrategy(
        ...     wall_eddy_diffusivity=0.001,
        ...     chamber_radius=0.5,
        ...     distribution_type="discrete",
        ... )
        >>> rate = strategy.rate(
        ...     particle=particle,
        ...     temperature=298.0,
        ...     pressure=101325.0,
        ... )
        >>> particle = strategy.step(
        ...     particle=particle,
        ...     temperature=298.0,
        ...     pressure=101325.0,
        ...     time_step=1.0,
        ... )
    """

    chamber_radius: float

    @validate_inputs(
        {
            "wall_eddy_diffusivity": "positive",
            "chamber_radius": "positive",
        }
    )
    def __init__(
        self,
        wall_eddy_diffusivity: float,
        chamber_radius: float,
        distribution_type: str = "discrete",
    ) -> None:
        """Initialize spherical wall loss strategy.

        Args:
            wall_eddy_diffusivity: Wall eddy diffusivity [1/s].
            chamber_radius: Radius of the spherical chamber [m].
            distribution_type: Distribution type ("discrete",
                "continuous_pdf", or "particle_resolved").
        """
        super().__init__(
            wall_eddy_diffusivity=wall_eddy_diffusivity,
            distribution_type=distribution_type,
        )
        self.chamber_radius = chamber_radius

    def loss_coefficient(
        self,
        particle: ParticleData,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Return the spherical wall loss coefficient for the given state.

        The coefficient is computed from the system state using
        :func:`get_spherical_wall_loss_coefficient_via_system_state`.

        Args:
            particle: Native particle data providing radius and density.
            temperature: Gas temperature [K].
            pressure: Gas pressure [Pa].

        Returns:
            Wall loss coefficient [1/s].
        """
        return _native.coefficient(self, particle, temperature, pressure)

    def loss_coefficient_for_particles(
        self,
        particle_radius: NDArray[np.float64],
        particle_density: NDArray[np.float64],
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Return the wall loss coefficient for given particle properties.

        This method is used for particle-resolved simulations where we
        need to calculate coefficients for a subset of active particles.

        Args:
            particle_radius: Particle radii [m].
            particle_density: Particle densities [kg/m^3].
            temperature: Gas temperature [K].
            pressure: Gas pressure [Pa].

        Returns:
            Wall loss coefficient [1/s] for each particle.
        """
        coefficient = get_spherical_wall_loss_coefficient_via_system_state(
            wall_eddy_diffusivity=self.wall_eddy_diffusivity,
            particle_radius=particle_radius,
            particle_density=particle_density,
            temperature=temperature,
            pressure=pressure,
            chamber_radius=self.chamber_radius,
        )
        return np.asarray(coefficient)


class ChargedWallLossStrategy(WallLossStrategy):
    """Wall loss strategy with electrostatic effects.

    Extends neutral wall loss with image-charge enhancement, optional
    electric-field drift, and diffusion modification for charged particles.
    Behaves as the neutral strategy when both charge and field terms are zero
    while still applying the image-charge-only enhancement when charge is
    non-zero and ``wall_potential`` is zero.

    References:
        McMurry, P. H., & Rader, D. J. (1985). Aerosol wall losses in
        electrically charged chambers. Aerosol Science and Technology, 4(3),
        249–268.
        Lai, A. C. K., & Nazaroff, W. W. (2000). Modeling indoor particle
        deposition from turbulent flow onto smooth surfaces. J. Aerosol Sci.,
        31(4), 463–476.
        Hinds, W. C. (1999). *Aerosol Technology*. Wiley.
    """

    chamber_geometry: str
    chamber_radius: Union[float, None]
    chamber_dimensions: Union[Tuple[float, float, float], None]
    wall_potential: float
    wall_electric_field: Union[float, Tuple[float, float, float]]

    @validate_inputs(
        {
            "wall_eddy_diffusivity": "positive",
            "wall_potential": "finite",
            "wall_electric_field": "finite",
            "chamber_radius": "positive",
            "chamber_dimensions": "positive",
        }
    )
    def __init__(
        self,
        wall_eddy_diffusivity: float,
        chamber_geometry: str,
        chamber_radius: Union[float, None] = None,
        chamber_dimensions: Union[Tuple[float, float, float], None] = None,
        wall_potential: float = 0.0,
        wall_electric_field: Union[float, Tuple[float, float, float]] = 0.0,
        distribution_type: str = "discrete",
    ) -> None:
        """Initialize charged wall loss strategy.

        Args:
            wall_eddy_diffusivity: Wall eddy diffusivity [1/s].
            chamber_geometry: Geometry string ("spherical" or "rectangular").
            chamber_radius: Radius for spherical chambers [m].
            chamber_dimensions: Dimensions (length, width, height) for
                rectangular chambers [m].
            wall_potential: Wall potential [V]; zero keeps image-charge term.
            wall_electric_field: Electric field magnitude [V/m] (scalar) or
                3-vector for rectangular chambers. Zero disables drift term.
            distribution_type: Distribution type.

        Raises:
            ValueError: If geometry is invalid or required geometry parameters
                are missing.
        """
        super().__init__(
            wall_eddy_diffusivity=wall_eddy_diffusivity,
            distribution_type=distribution_type,
        )
        geometry = chamber_geometry.lower()
        if geometry not in {"spherical", "rectangular"}:
            raise ValueError(
                "chamber_geometry must be 'spherical' or 'rectangular'."
            )
        if geometry == "spherical":
            if chamber_radius is None:
                raise ValueError("chamber_radius is required for spherical.")
            self.chamber_radius = float(chamber_radius)
            self.chamber_dimensions = None
        else:
            if chamber_dimensions is None:
                raise ValueError(
                    "chamber_dimensions are required for rectangular."
                )
            if len(chamber_dimensions) != 3:
                raise ValueError(
                    "chamber_dimensions must be length, width, height."
                )
            if any(dimension <= 0 for dimension in chamber_dimensions):
                raise ValueError("All chamber dimensions must be positive")
            self.chamber_dimensions = (
                float(chamber_dimensions[0]),
                float(chamber_dimensions[1]),
                float(chamber_dimensions[2]),
            )
            self.chamber_radius = None
        self.chamber_geometry = geometry
        self.wall_potential = float(wall_potential)
        self.wall_electric_field = wall_electric_field

    @property
    def _geometry_scale(self) -> float:
        """Return characteristic geometry length for field scaling."""
        if self.chamber_geometry == "spherical" and self.chamber_radius:
            return self.chamber_radius
        if self.chamber_dimensions is not None:
            return float(np.min(self.chamber_dimensions))
        return 1.0

    def _neutral_coefficient(
        self,
        particle_radius: NDArray[np.float64],
        particle_density: NDArray[np.float64],
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Return neutral wall loss coefficient for configured geometry.

        Args:
            particle_radius: Particle radii in meters.
            particle_density: Particle densities in kg/m³.
            temperature: Gas temperature in kelvin.
            pressure: Gas pressure in pascals.

        Returns:
            Neutral wall loss coefficient in 1/s for each particle size.
        """
        if self.chamber_geometry == "spherical":
            return np.asarray(
                get_spherical_wall_loss_coefficient_via_system_state(
                    wall_eddy_diffusivity=self.wall_eddy_diffusivity,
                    particle_radius=particle_radius,
                    particle_density=particle_density,
                    temperature=temperature,
                    pressure=pressure,
                    chamber_radius=float(cast(float, self.chamber_radius)),
                )
            )
        return np.asarray(
            get_rectangle_wall_loss_coefficient_via_system_state(
                wall_eddy_diffusivity=self.wall_eddy_diffusivity,
                particle_radius=particle_radius,
                particle_density=particle_density,
                temperature=temperature,
                pressure=pressure,
                chamber_dimensions=cast(
                    Tuple[float, float, float], self.chamber_dimensions
                ),
            )
        )

    def _electrostatic_factor(
        self,
        particle_radius: NDArray[np.float64],
        particle_charge: NDArray[np.float64],
        temperature: float,
    ) -> NDArray[np.float64]:
        """Compute electrostatic enhancement factor.

        Applies image-charge-derived enhancement based on particle charge.
        Returns ones when charge is zero so the strategy reduces to neutral
        even if ``wall_potential`` is zero.

        Args:
            particle_radius: Particle radii in meters.
            particle_charge: Particle charge in elementary charges.
            temperature: Gas temperature in kelvin.

        Returns:
            Multiplicative electrostatic factor for the wall loss coefficient.
        """
        if not np.any(particle_charge):
            return np.ones_like(particle_radius, dtype=np.float64)
        phi_raw = np.asarray(
            get_coulomb_enhancement_ratio(
                particle_radius=particle_radius,
                charge=particle_charge,
                temperature=temperature,
            )
        )
        phi = phi_raw.diagonal() if phi_raw.ndim == 2 else phi_raw
        phi = np.abs(phi)
        phi_clipped = np.clip(phi, -50.0, 50.0)
        factor = np.exp(phi_clipped)
        return np.where(particle_charge == 0, 1.0, factor)

    def _resolve_electric_field(self) -> float:
        """Return resolved electric field magnitude in V/m.

        Combines the configured wall electric field (scalar or vector) with a
        geometry-scaled potential-derived field; returns zero when both are
        zero.

        Returns:
            Total electric field magnitude in volts per meter.
        """
        if isinstance(self.wall_electric_field, (tuple, list, np.ndarray)):
            magnitude = float(np.linalg.norm(self.wall_electric_field))
        else:
            magnitude = float(self.wall_electric_field)
        potential_field = 0.0
        if self.wall_potential != 0 and self._geometry_scale > 0:
            potential_field = self.wall_potential / self._geometry_scale
        return magnitude + potential_field

    def _drift_term(
        self,
        particle_radius: NDArray[np.float64],
        particle_charge: NDArray[np.float64],
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Compute electric-field drift contribution.

        Returns zero when charge is zero or when the resolved electric field
        is zero.

        Args:
            particle_radius: Particle radii in meters.
            particle_charge: Particle charge in elementary charges.
            temperature: Gas temperature in kelvin.
            pressure: Gas pressure in pascals.

        Returns:
            Additive drift term in 1/s for the wall loss coefficient.
        """
        del pressure  # pressure currently unused for drift approximation
        if not np.any(particle_charge):
            return np.zeros_like(particle_radius, dtype=np.float64)
        electric_field = self._resolve_electric_field()
        if electric_field == 0:
            return np.zeros_like(particle_radius, dtype=np.float64)
        viscosity = get_dynamic_viscosity(temperature=temperature)
        diameter = 2.0 * np.clip(particle_radius, 1e-30, None)
        mobility = (
            np.abs(particle_charge)
            * ELEMENTARY_CHARGE_VALUE
            / (3.0 * np.pi * viscosity * diameter)
        )
        drift_velocity = mobility * electric_field * np.sign(particle_charge)
        drift_term = drift_velocity / max(self._geometry_scale, 1e-30)
        return np.nan_to_num(drift_term, nan=0.0)

    def _combine_coefficients(
        self,
        neutral: NDArray[np.float64],
        electrostatic_factor: NDArray[np.float64],
        drift_term: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Combine neutral, electrostatic, and drift contributions.

        Args:
            neutral: Neutral wall loss coefficient in 1/s.
            electrostatic_factor: Multiplicative factor from image charge.
            drift_term: Additive drift term in 1/s.

        Returns:
            Combined wall loss coefficient clipped to non-negative finite
            values.
        """
        combined = neutral * electrostatic_factor + drift_term
        return np.clip(
            np.nan_to_num(combined, nan=0.0),
            0.0,
            np.finfo(np.float64).max,
        )

    def loss_coefficient(
        self,
        particle: ParticleData,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Compute charged wall loss coefficient from particle properties.

        Applies image-charge enhancement even when ``wall_potential`` is zero
        and adds drift when ``wall_electric_field`` is non-zero.
        Reduces to the neutral coefficient when charge and field are zero and
        supports particle-resolved active subsets.

        Args:
            particle: Native particle data providing radius, density, and
                charge.
            temperature: Gas temperature in kelvin.
            pressure: Gas pressure in pascals.

        Returns:
            Wall loss coefficient in 1/s as a scalar or array.
        """
        return _native.coefficient(self, particle, temperature, pressure)

    def loss_coefficient_for_particles(
        self,
        particle_radius: NDArray[np.float64],
        particle_density: NDArray[np.float64],
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Compute charged wall loss coefficient for provided particle arrays.

        This charge-free interface assumes neutral particles. For charged
        material, use ``compute_coefficient_from_arrays`` with explicit charge
        arrays. No earlier particle evaluation supplies implicit charge state.

        Args:
            particle_radius: Particle radii in meters.
            particle_density: Particle densities in kg/m³.
            temperature: Gas temperature in kelvin.
            pressure: Gas pressure in pascals.

        Returns:
            Wall loss coefficient in 1/s for each particle.
        """
        charge_array = np.zeros_like(particle_radius, dtype=np.float64)
        neutral = self._neutral_coefficient(
            particle_radius=particle_radius,
            particle_density=particle_density,
            temperature=temperature,
            pressure=pressure,
        )
        electrostatic_factor = self._electrostatic_factor(
            particle_radius=particle_radius,
            particle_charge=charge_array,
            temperature=temperature,
        )
        drift_term = self._drift_term(
            particle_radius=particle_radius,
            particle_charge=charge_array,
            temperature=temperature,
            pressure=pressure,
        )
        return self._combine_coefficients(
            neutral=neutral,
            electrostatic_factor=electrostatic_factor,
            drift_term=drift_term,
        )

    def compute_coefficient_from_arrays(
        self,
        particle_radius: NDArray[np.float64],
        particle_density: NDArray[np.float64],
        particle_charge: NDArray[np.float64],
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Compute charged wall loss coefficient from explicit arrays.

        Public method for computing coefficients with explicit charge arrays,
        useful for rate helpers and direct calculations without cached state.

        Args:
            particle_radius: Particle radii in meters.
            particle_density: Particle densities in kg/m³.
            particle_charge: Particle charge in elementary charges.
            temperature: Gas temperature in kelvin.
            pressure: Gas pressure in pascals.

        Returns:
            Wall loss coefficient in 1/s for each particle.
        """
        neutral = self._neutral_coefficient(
            particle_radius=particle_radius,
            particle_density=particle_density,
            temperature=temperature,
            pressure=pressure,
        )
        electrostatic_factor = self._electrostatic_factor(
            particle_radius=particle_radius,
            particle_charge=particle_charge,
            temperature=temperature,
        )
        drift_term = self._drift_term(
            particle_radius=particle_radius,
            particle_charge=particle_charge,
            temperature=temperature,
            pressure=pressure,
        )
        return self._combine_coefficients(
            neutral=neutral,
            electrostatic_factor=electrostatic_factor,
            drift_term=drift_term,
        )

    def step(
        self,
        particle: ParticleData,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> ParticleData:
        """Advance particle-resolved concentration with charged wall loss.

        Computes survival probabilities using neutral, image-charge, and
        optional drift contributions. Clamps probabilities to [0, 1] and
        updates particle concentrations and distributions.

        Args:
            particle: Native particle data to update.
            temperature: Gas temperature in kelvin.
            pressure: Gas pressure in pascals.
            time_step: Time step in seconds.

        Returns:
            Updated native particle data.
        """
        return _native.step(self, particle, temperature, pressure, time_step)


class RectangularWallLossStrategy(WallLossStrategy):
    """Wall loss strategy for rectangular (box) chambers.

    Calculates particle wall deposition in rectangular chamber geometry
    using turbulent diffusion and gravitational settling. Supports
    discrete, continuous PDF, and particle-resolved distributions.

    Attributes:
        wall_eddy_diffusivity: Wall eddy diffusivity [1/s].
        chamber_dimensions: Chamber dimensions (length, width, height) [m].
        distribution_type: Distribution type ("discrete",
            "continuous_pdf", or "particle_resolved").

    Examples:
        >>> import particula as par
        >>> particle = par.particles.PresetParticleRadiusBuilder().build()
        >>> strategy = par.dynamics.RectangularWallLossStrategy(
        ...     wall_eddy_diffusivity=0.001,
        ...     chamber_dimensions=(1.0, 0.5, 0.5),
        ...     distribution_type="discrete",
        ... )
        >>> rate = strategy.rate(
        ...     particle=particle,
        ...     temperature=298.0,
        ...     pressure=101325.0,
        ... )
        >>> _ = strategy.step(
        ...     particle=particle,
        ...     temperature=298.0,
        ...     pressure=101325.0,
        ...     time_step=1.0,
        ... )

    References:
        Crump, J. G., & Seinfeld, J. H. (1981). Turbulent deposition and
        gravitational sedimentation of an aerosol in a vessel of arbitrary
        shape. Journal of Aerosol Science, 12(5), 405–415.
        McMurry, P. H., & Rader, D. J. (1985). Aerosol wall losses in
        electrically charged chambers. Aerosol Science and Technology, 4(3),
        249–268.
    """

    chamber_dimensions: Tuple[float, float, float]

    @validate_inputs({"wall_eddy_diffusivity": "positive"})
    def __init__(
        self,
        wall_eddy_diffusivity: float,
        chamber_dimensions: Tuple[float, float, float],
        distribution_type: str = "discrete",
    ) -> None:
        """Initialize rectangular wall loss strategy.

        Args:
            wall_eddy_diffusivity: Wall eddy diffusivity [1/s].
            chamber_dimensions: Chamber dimensions (length, width, height)
                in meters. All must be positive.
            distribution_type: Distribution type ("discrete",
                "continuous_pdf", or "particle_resolved").

        Raises:
            ValueError: If ``chamber_dimensions`` does not contain exactly
                three positive values.
            ValueError: If ``distribution_type`` is not supported.
        """
        super().__init__(
            wall_eddy_diffusivity=wall_eddy_diffusivity,
            distribution_type=distribution_type,
        )
        if len(chamber_dimensions) != 3:
            raise ValueError(
                "chamber_dimensions must be a tuple of length, width, height"
            )
        if any(dimension <= 0 for dimension in chamber_dimensions):
            raise ValueError("All chamber dimensions must be positive")
        self.chamber_dimensions = (
            float(chamber_dimensions[0]),
            float(chamber_dimensions[1]),
            float(chamber_dimensions[2]),
        )

    def loss_coefficient(
        self,
        particle: ParticleData,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Return the rectangular wall loss coefficient for the state.

        Args:
            particle: Native particle data providing radius and density.
            temperature: Gas temperature [K].
            pressure: Gas pressure [Pa].

        Returns:
            Wall loss coefficient [1/s] for each particle bin.
        """
        return _native.coefficient(self, particle, temperature, pressure)

    def loss_coefficient_for_particles(
        self,
        particle_radius: NDArray[np.float64],
        particle_density: NDArray[np.float64],
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Return wall loss coefficient for provided particle properties.

        This method is used for particle-resolved simulations to evaluate
        coefficients on active particles without reconstructing the full
        representation.

        Args:
            particle_radius: Particle radii [m].
            particle_density: Particle densities [kg/m^3].
            temperature: Gas temperature [K].
            pressure: Gas pressure [Pa].

        Returns:
            Wall loss coefficient [1/s] for each particle.
        """
        coefficient = get_rectangle_wall_loss_coefficient_via_system_state(
            wall_eddy_diffusivity=self.wall_eddy_diffusivity,
            particle_radius=particle_radius,
            particle_density=particle_density,
            temperature=temperature,
            pressure=pressure,
            chamber_dimensions=self.chamber_dimensions,
        )
        return np.asarray(coefficient)
