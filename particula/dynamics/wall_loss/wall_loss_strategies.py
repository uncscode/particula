"""Wall loss strategy abstractions and implementations.

Defines abstract and concrete strategies for modeling particle wall
loss processes in different chamber geometries. Strategies operate on
:class:`~particula.particles.representation.ParticleRepresentation`
objects and support multiple distribution types. Implementations are
provided for spherical and rectangular chambers.

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
from typing import Callable, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from particula.dynamics.properties.wall_loss_coefficient import (
    get_rectangle_wall_loss_coefficient_via_system_state,
    get_spherical_wall_loss_coefficient_via_system_state,
)
from particula.particles.representation import ParticleRepresentation
from particula.util.validate_inputs import validate_inputs


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
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Return the wall loss coefficient for the given state.

        Args:
            particle: Particle representation to evaluate.
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
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Return the wall loss rate for the given state.

        The loss rate is computed as ``-k(Dp) * c`` where ``k(Dp)`` is the
        size-dependent wall loss coefficient and ``c`` is the particle number
        concentration.

        Args:
            particle: Particle representation to evaluate.
            temperature: Gas temperature [K].
            pressure: Gas pressure [Pa].

        Returns:
            Wall loss rate [#/m^3 s] (negative values indicate loss).
        """
        coefficient = self.loss_coefficient(
            particle=particle,
            temperature=temperature,
            pressure=pressure,
        )
        concentration = particle.get_concentration()
        return -np.asarray(coefficient) * np.asarray(concentration)

    def rate(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Return the wall loss rate as an array.

        Args:
            particle: Particle representation to evaluate.
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
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> ParticleRepresentation:
        """Advance the particle representation by one wall loss step.

        For ``"discrete"`` and ``"continuous_pdf"`` distributions, this
        applies a deterministic first-order loss to the number
        concentration in each bin. For ``"particle_resolved"``, it
        applies a stochastic removal process using binomial random draws
        based on survival probability.

        Args:
            particle: Particle representation to update.
            temperature: Gas temperature [K].
            pressure: Gas pressure [Pa].
            time_step: Time step [s].

        Returns:
            Updated particle representation (same instance).

        Raises:
            ValueError: If ``distribution_type`` is not supported.
        """
        if self.distribution_type in {"discrete", "continuous_pdf"}:
            rate = self.rate(
                particle=particle,
                temperature=temperature,
                pressure=pressure,
            )
            particle.add_concentration(rate * time_step)
            return particle

        if self.distribution_type == "particle_resolved":
            # For particle-resolved representations, use stochastic removal
            concentration = np.asarray(particle.get_concentration())
            radius = np.asarray(particle.get_radius())
            density = np.asarray(particle.get_effective_density())

            # Create a closure that binds temperature and pressure
            def coeff_func(r, d):
                return self.loss_coefficient_for_particles(
                    r, d, temperature, pressure
                )

            # Use the helper function to determine which particles survive
            survived = get_particle_resolved_wall_loss_step(
                particle_radius=radius,
                particle_density=density,
                concentration=concentration,
                loss_coefficient_func=coeff_func,
                time_step=time_step,
                random_generator=self.random_generator,
            )

            # Update concentration: particles either survive (1) or are lost (0)
            new_concentration = concentration * survived

            # Map back to the internal concentration stored on the
            # representation, which is defined per representation volume.
            volume = particle.get_volume()
            particle.concentration = new_concentration * volume

            # Set particle mass to zero for particles that were lost
            lost_particles = (concentration > 0) & (survived == 0)
            if np.any(lost_particles):
                if particle.distribution.ndim == 1:
                    # 1D array: single species per particle
                    particle.distribution[lost_particles] = 0.0
                else:
                    # 2D matrix: multiple species per particle
                    particle.distribution[lost_particles, :] = 0.0

            return particle

        raise ValueError(
            "Invalid distribution type. Must be one of 'discrete', "
            + "'continuous_pdf', or 'particle_resolved'."
        )


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
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Return the spherical wall loss coefficient for the given state.

        The coefficient is computed from the system state using
        :func:`get_spherical_wall_loss_coefficient_via_system_state`.

        Args:
            particle: Particle representation providing radius and density.
            temperature: Gas temperature [K].
            pressure: Gas pressure [Pa].

        Returns:
            Wall loss coefficient [1/s].
        """
        return get_spherical_wall_loss_coefficient_via_system_state(
            wall_eddy_diffusivity=self.wall_eddy_diffusivity,
            particle_radius=particle.get_radius(),
            particle_density=particle.get_effective_density(),
            temperature=temperature,
            pressure=pressure,
            chamber_radius=self.chamber_radius,
        )

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
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Return the rectangular wall loss coefficient for the state.

        Args:
            particle: Particle representation providing radius and density.
            temperature: Gas temperature [K].
            pressure: Gas pressure [Pa].

        Returns:
            Wall loss coefficient [1/s] for each particle bin.
        """
        return get_rectangle_wall_loss_coefficient_via_system_state(
            wall_eddy_diffusivity=self.wall_eddy_diffusivity,
            particle_radius=particle.get_radius(),
            particle_density=particle.get_effective_density(),
            temperature=temperature,
            pressure=pressure,
            chamber_dimensions=self.chamber_dimensions,
        )

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
