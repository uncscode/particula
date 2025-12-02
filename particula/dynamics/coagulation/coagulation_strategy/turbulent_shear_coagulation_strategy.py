"""Turbulent Shear coagulation strategies and calculations.

Provides turbulence-based coagulation kernels following Saffman & Turner
(1956). This module contains classes and functions for computing the
turbulent shear coagulation rate in aerosol systems.

Classes:
    - TurbulentShearCoagulationStrategy : Implements the abstract base
      class for coagulation using a turbulent shear kernel.
"""

import logging
from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.dynamics.coagulation.turbulent_shear_kernel import (
    get_turbulent_shear_kernel_st1956_via_system_state,
)
from particula.particles.representation import ParticleRepresentation

from .coagulation_strategy_abc import CoagulationStrategyABC

logger = logging.getLogger("particula")


class TurbulentShearCoagulationStrategy(CoagulationStrategyABC):
    """Turbulent shear coagulation strategy for aerosol particles.

    Implements the Saffman & Turner (1956) turbulent shear coagulation kernel,
    extending the base `CoagulationStrategyABC` class to provide a physically
    consistent model of coagulation in turbulent flow.

    Attributes:
        - distribution_type : The type of particle distribution for coagulation
          ("discrete", "continuous_pdf", or "particle_resolved").
        - turbulent_dissipation : Turbulent kinetic energy dissipation
          rate [m^2/s^3].
        - fluid_density : Fluid density [kg/m^3].

    Methods:
    - set_turbulent_dissipation : Set the turbulent kinetic energy dissipation
      rate.
    - dimensionless_kernel : (Not implemented here) Raise NotImplementedError.
    - kernel : Compute the turbulent shear coagulation kernel via
      Saffman-Turner approach.
    - loss_rate : Calculate the coagulation loss rate.
    - gain_rate : Calculate the coagulation gain rate.
    - net_rate : Get the net coagulation rate (gain - loss).
    - step : Perform a single step of coagulation.
    - diffusive_knudsen : Calculate the diffusive Knudsen number.
    - coulomb_potential_ratio : Compute Coulomb potential ratio.
    - friction_factor : Compute the effective friction factor.

    Examples:
        ```py title="Example usage of TurbulentShearCoagulationStrategy"
        import particula as par
        strategy = par.dynamics.TurbulentShearCoagulationStrategy(
            distribution_type="discrete",
            turbulent_dissipation=0.01,
            fluid_density=1.225,
        )
        # Use strategy.kernel(...) to get the coagulation kernel
        ```

    References:
        - Saffman, P. G., & Turner, J. S. (1956). On the collision of drops in
          turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.
          https://doi.org/10.1017/S0022112056000020
    """

    def __init__(
        self,
        distribution_type: str,
        turbulent_dissipation: float,
        fluid_density: float,
    ):
        """Initialize the turbulent shear coagulation strategy.

        Arguments:
            - distribution_type : The distribution type ("discrete",
              "continuous_pdf", or "particle_resolved").
            - turbulent_dissipation : Turbulent kinetic energy dissipation
              rate [m^2/s^3].
            - fluid_density : The fluid density [kg/m^3].

        Returns:
            - None
        """
        super().__init__(distribution_type=distribution_type)
        self.turbulent_dissipation = turbulent_dissipation
        self.fluid_density = fluid_density

    def set_turbulent_dissipation(self, turbulent_dissipation: float):
        """Set the turbulent kinetic energy dissipation rate.

        Arguments:
            - turbulent_dissipation : Turbulent kinetic energy dissipation
              rate [m^2/s^3].

        Returns:
            - Self (TurbulentShearCoagulationStrategy)

        Examples:
            ```py
            strategy.set_turbulent_dissipation(0.02)
            ```
        """
        self.turbulent_dissipation = turbulent_dissipation
        return self

    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute a dimensionless kernel (H).

        Not implemented for turbulent shear; raises NotImplementedError.

        Arguments:
            - diffusive_knudsen : The diffusive Knudsen number [dimensionless].
            - coulomb_potential_ratio : The Coulomb potential ratio
              [dimensionless].

        Returns:
            - NDArray[np.float64] : Not returned; raises error instead.

        Examples:
            ```py
            # This method is not supported here
            try:
                result = strategy.dimensionless_kernel(diff_kn, phi_ratio)
            except NotImplementedError:
                print("Not implemented for turbulent shear strategy.")
            ```

        References:
            - Saffman & Turner (1956) used dimensional forms; dimensionless
              form is not covered.
        """
        message = (
            "Dimensionless kernel not implemented in turbulent shear "
            + "coagulation strategy."
        )
        logger.error(message)
        raise NotImplementedError(message)

    def kernel(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Compute the dimensioned turbulent shear coagulation kernel [m^3/s].

        Uses the system state to calculate the Saffman-Turner (1956) kernel,
        which depends on the dissipation rate of turbulent kinetic energy,
        fluid density, and particle radius.

        Arguments:
            - particle : The ParticleRepresentation instance to retrieve
              particle radii.
            - temperature : The system temperature [K].
            - pressure : The system pressure [Pa].

        Returns:
            - float or NDArray[np.float64] : The coagulation kernel(s) [m^3/s].

        Examples:
            ```py title="Example usage of kernel method"
            kernel_value = strategy.kernel(
                particle=ParticleRepresentation(...),
                temperature=298.15,
                pressure=101325
            )
            # kernel_value could be a float or array depending on the
            # particle representation
            ```

        References:
            - Saffman, P. G., & Turner, J. S. (1956). On the collision of drops
              in turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.
              https://doi.org/10.1017/S0022112056000020
        """
        return get_turbulent_shear_kernel_st1956_via_system_state(
            particle_radius=particle.get_radius(),
            turbulent_dissipation=self.turbulent_dissipation,
            temperature=temperature,
            fluid_density=self.fluid_density,
        )
