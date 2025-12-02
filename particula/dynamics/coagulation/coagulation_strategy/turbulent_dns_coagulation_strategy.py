"""Turbulent DNS coagulation strategies for particles above 1 µm.

Implements the DNS-based coagulation kernel from Ayala et al. (2008) and
adjusted for typical atmospheric or industrial conditions.
Provides classes and methods to compute collision rates under turbulent
dissipation using direct numerical simulation (DNS) approaches.
"""

import logging
from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.particles.representation import ParticleRepresentation

from ..turbulent_dns_kernel.turbulent_dns_kernel_ao2008 import (
    get_turbulent_dns_kernel_ao2008_via_system_state,
)
from .coagulation_strategy_abc import CoagulationStrategyABC

logger = logging.getLogger("particula")


class TurbulentDNSCoagulationStrategy(CoagulationStrategyABC):
    """Turbulent DNS coagulation strategy for aerosols.

    Implements methods from `CoagulationStrategyABC`, applying the
    turbulent DNS kernel following Ayala et al. (2008). Suitable for
    coagulation of particles larger than 1 µm in turbulent flow fields.

    Attributes:
        - distribution_type : The particle distribution type ("discrete",
          "continuous_pdf", or "particle_resolved").
        - turbulent_dissipation : Turbulent kinetic energy dissipation
          [m^2/s^3] used in DNS fits (examples: 0.001, 0.01, 0.04).
        - fluid_density : The fluid (air) density [kg/m^3].
        - reynolds_lambda : Reynolds lambda of air (e.g., 23 or 74).
        - relative_velocity : Relative velocity of the air [m/s] for
          collisions.

    Methods:
    - set_turbulent_dissipation : Change turbulent dissipation rate.
    - set_reynolds_lambda : Update the Reynolds lambda.
    - set_relative_velocity : Update the relative velocity.
    - dimensionless_kernel : Raise NotImplementedError for DNS approach.
    - kernel : Return the DNS-based coagulation kernel.
    - loss_rate : Calculate the coagulation loss rate.
    - gain_rate : Calculate the coagulation gain rate.
    - net_rate : Get the net coagulation rate (gain - loss).
    - step : Perform a single step of coagulation.
    - diffusive_knudsen : Calculate the diffusive Knudsen number.
    - coulomb_potential_ratio : Compute Coulomb potential ratio.
    - friction_factor : Compute the effective friction factor.

    Examples:
        ```py title="Example usage of TurbulentDNSCoagulationStrategy"
        import particula as par
        strategy = par.dynamics.TurbulentDNSCoagulationStrategy(
            distribution_type="discrete",
            turbulent_dissipation=0.01,
            fluid_density=1.225,
            reynolds_lambda=23,
            relative_velocity=0.5
        )
        # Use strategy.kernel(...) to compute the DNS-based kernel
        ```

    References:
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on the
      geometric collision rate of sedimenting droplets. Part 2. Theory and
      parameterization. New Journal of Physics, 10.
      [DOI](https://doi.org/10.1088/1367-2630/10/7/075016)
    """

    def __init__(
        self,
        distribution_type: str,
        turbulent_dissipation: float,
        fluid_density: float,
        reynolds_lambda: float,
        relative_velocity: float,
    ):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Initialize the TurbulentDNSCoagulationStrategy.

        Arguments:
            - distribution_type : The distribution type ("discrete",
              "continuous_pdf", or "particle_resolved").
            - turbulent_dissipation : Turbulent kinetic energy dissipation
              [m^2/s^3].
            - fluid_density : The fluid density [kg/m^3].
            - reynolds_lambda : Reynolds lambda or characteristic Reynolds
              number.
            - relative_velocity : Relative velocity of the flow [m/s].

        Returns:
            - None
        """
        super().__init__(distribution_type=distribution_type)
        self.turbulent_dissipation = turbulent_dissipation
        self.fluid_density = fluid_density
        self.reynolds_lambda = reynolds_lambda
        self.relative_velocity = relative_velocity

    def set_turbulent_dissipation(self, turbulent_dissipation: float):
        """Set the turbulent kinetic energy dissipation rate.

        Arguments:
            - turbulent_dissipation : Turbulent dissipation [m^2/s^3].

        Returns:
            - TurbulentDNSCoagulationStrategy : Self, allowing method chaining.

        Examples:
            ```py
            strategy.set_turbulent_dissipation(0.02)
            ```
        """
        self.turbulent_dissipation = turbulent_dissipation
        return self

    def set_reynolds_lambda(self, reynolds_lambda: float):
        """Set the Reynolds lambda value.

        Arguments:
            - reynolds_lambda : Reynolds lambda [dimensionless].

        Returns:
            - TurbulentDNSCoagulationStrategy : Self, for method chaining.

        Examples:
            ```py
            strategy.set_reynolds_lambda(74)
            ```
        """
        self.reynolds_lambda = reynolds_lambda
        return self

    def set_relative_velocity(self, relative_velocity: float):
        """Set the relative velocity of the flow [m/s].

        Arguments:
            - relative_velocity : Relative velocity in [m/s].

        Returns:
            - TurbulentDNSCoagulationStrategy : Self, for method chaining.

        Examples:
            ```py
            strategy.set_relative_velocity(0.8)
            ```
        """
        self.relative_velocity = relative_velocity
        return self

    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute or return the dimensionless kernel (H).

        Not implemented for DNS-based approaches, so raises
        NotImplementedError.

        Arguments:
            - diffusive_knudsen : The diffusive Knudsen number(s)
              [dimensionless].
            - coulomb_potential_ratio : The Coulomb potential ratio(s)
              [dimensionless].

        Returns:
            - None : Raises NotImplementedError instead.

        Raises:
            - NotImplementedError : This strategy does not support
            dimensionless kernels.
        """
        message = (
            "The dimensionless kernel is not implemented for "
            "TurbulentDNSCoagulationStrategy."
        )
        logger.error(message)
        raise NotImplementedError(message)

    def kernel(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Compute the DNS-based coagulation kernel [m^3/s].

        Uses the `get_turbulent_dns_kernel_ao2008_via_system_state` function to
        calculate collision rates following Ayala et al. (2008). This approach
        accounts for turbulent dissipation, fluid density, Reynolds lambda,
        and relative velocity.

        Arguments:
            - particle : The ParticleRepresentation whose radii and density
              are needed.
            - temperature : The temperature of the system [K].
            - pressure : The system pressure [Pa] (unused here, but included
              for interface consistency).

        Returns:
            - The DNS-based coagulation kernel(s).

        Examples:
            ```py
            kernel_values = strategy.kernel(
                particle=my_particle,
                temperature=298.15,
                pressure=101325
            )
            # kernel_values may be a float or array, depending on the
            # distribution
            ```

        References:
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence
          on the geometric collision rate of sedimenting droplets. Part 2.
          New Journal of Physics, 10.
          [DOI](https://doi.org/10.1088/1367-2630/10/7/075016)
        """
        return get_turbulent_dns_kernel_ao2008_via_system_state(
            particle_radius=particle.get_radius(),
            particle_density=particle.get_mean_effective_density(),
            fluid_density=self.fluid_density,
            turbulent_dissipation=self.turbulent_dissipation,
            re_lambda=self.reynolds_lambda,
            relative_velocity=self.relative_velocity,
            temperature=temperature,
        )
