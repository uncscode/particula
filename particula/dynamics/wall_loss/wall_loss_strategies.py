"""Wall loss strategy abstractions and implementations.

Defines abstract and concrete strategies for modeling particle wall
loss processes in different chamber geometries. Strategies operate
on :class:`~particula.particles.representation.ParticleRepresentation`
objects and support multiple distribution types.

The wall loss rate is modeled as a first-order loss process

.. math::

    L = -k c,

where :math:`L` is the wall loss rate [#/m^3 s], :math:`k` is the wall
loss coefficient [1/s], and :math:`c` is the particle number
concentration [#/m^3].

References:
    Crump, J. G., & Seinfeld, J. H. (1981). Turbulent deposition and
    gravitational sedimentation of an aerosol in a vessel of arbitrary
    shape. Journal of Aerosol Science, 12(5), 405–415.
"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.dynamics.properties.wall_loss_coefficient import (
    get_spherical_wall_loss_coefficient_via_system_state,
)
from particula.particles.representation import ParticleRepresentation
from particula.util.validate_inputs import validate_inputs


class WallLossStrategy(ABC):
    """Abstract base class for wall loss strategies.

    Wall loss strategies compute wall loss coefficients and associated
    rates for different distribution types.

    Attributes:
        wall_eddy_diffusivity: Wall eddy diffusivity [m^2/s].
        distribution_type: Distribution type ("discrete",
            "continuous_pdf", or "particle_resolved").
    """

    wall_eddy_diffusivity: float
    distribution_type: str

    @validate_inputs({"wall_eddy_diffusivity": "positive"})
    def __init__(
        self, wall_eddy_diffusivity: float, distribution_type: str = "discrete"
    ) -> None:
        """Initialize the wall loss strategy.

        Args:
            wall_eddy_diffusivity: Wall eddy diffusivity [m^2/s].
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

    def loss_rate(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Return the wall loss rate for the given state.

        The loss rate is computed as ``-k * c`` where ``k`` is the wall
        loss coefficient and ``c`` is the particle number concentration.

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
        applies a deterministic approximation to a stochastic removal
        process using a survival probability.

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
        rate = self.rate(
            particle=particle,
            temperature=temperature,
            pressure=pressure,
        )

        if self.distribution_type in {"discrete", "continuous_pdf"}:
            particle.add_concentration(rate * time_step)
            return particle

        if self.distribution_type == "particle_resolved":
            # For particle-resolved representations, update the underlying
            # concentration directly instead of going through the distribution
            # strategy. The particle-resolved distribution strategy enforces
            # that added concentrations are either all ones or all equal,
            # which is not compatible with a deterministic survival
            # probability update.
            concentration = np.asarray(particle.get_concentration())
            # rate is negative for loss; exp(rate * dt) gives survival
            survival_probability = np.exp(rate * time_step)
            # clamp survival probability to [0, 1] for numerical safety
            survival_probability = np.clip(survival_probability, 0.0, 1.0)
            new_concentration = concentration * survival_probability

            # Map back to the internal concentration stored on the
            # representation, which is defined per representation volume.
            volume = particle.get_volume()
            particle.concentration = new_concentration * volume
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
        wall_eddy_diffusivity: Wall eddy diffusivity [m^2/s].
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
            wall_eddy_diffusivity: Wall eddy diffusivity [m^2/s].
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
