"""Wall loss strategy module."""

from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

from particula.particles.representation import ParticleRepresentation
from particula.dynamics import wall_loss


class WallLossStrategy:
    """Strategy class for chamber wall loss calculations."""

    def __init__(
        self,
        wall_eddy_diffusivity: float,
        chamber_radius: Optional[float] = None,
        chamber_dimensions: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        if chamber_radius is None and chamber_dimensions is None:
            raise ValueError(
                "Either chamber_radius or chamber_dimensions must be provided."
            )
        self.wall_eddy_diffusivity = wall_eddy_diffusivity
        self.chamber_radius = chamber_radius
        self.chamber_dimensions = chamber_dimensions

    def rate(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Return the wall loss rate for the particle population."""
        if self.chamber_radius is not None:
            return wall_loss.get_spherical_wall_loss_rate(
                wall_eddy_diffusivity=self.wall_eddy_diffusivity,
                particle_radius=particle.get_radius(),
                particle_density=particle.get_density(),
                particle_concentration=particle.get_concentration(),
                temperature=temperature,
                pressure=pressure,
                chamber_radius=self.chamber_radius,
            )
        return wall_loss.get_rectangle_wall_loss_rate(
            wall_eddy_diffusivity=self.wall_eddy_diffusivity,
            particle_radius=particle.get_radius(),
            particle_density=particle.get_density(),
            particle_concentration=particle.get_concentration(),
            temperature=temperature,
            pressure=pressure,
            chamber_dimensions=self.chamber_dimensions,  # type: ignore
        )

    def step(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> ParticleRepresentation:
        """Advance particle concentration by one time step."""
        rate = self.rate(
            particle=particle,
            temperature=temperature,
            pressure=pressure,
        )
        particle.add_concentration(rate * time_step)
        return particle
