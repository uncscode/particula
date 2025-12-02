"""Charged particle coagulation strategy."""

import logging
from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.dynamics.coagulation.charged_kernel_strategy import (
    ChargedKernelStrategyABC,
)
from particula.particles.representation import ParticleRepresentation
from particula.util.reduced_quantity import get_reduced_self_broadcast

from .coagulation_strategy_abc import CoagulationStrategyABC

logger = logging.getLogger("particula")


class ChargedCoagulationStrategy(CoagulationStrategyABC):
    """Charged Brownian coagulation strategy using a dimensionless kernel.

    This class implements the methods defined in the CoagulationStrategyABC
    abstract class. A ChargedKernelStrategyABC instance is passed to define
    how the dimensionless kernel is calculated. This approach allows flexible
    handling of Coulomb interactions under various regimes.

    Attributes:
        kernel_strategy : Instance of ChargedKernelStrategyABC used to
        calculate dimensionless and dimensioned kernels.

    Methods:
    - dimensionless_kernel : Compute dimensionless kernel values for
      charged coagulation.
    - kernel : Convert dimensionless kernel values into a dimensioned
      coagulation kernel.
    - loss_rate : Calculate the coagulation loss rate.
    - gain_rate : Calculate the coagulation gain rate.
    - net_rate : Get the net coagulation rate (gain - loss).
    - step : Perform a single step of coagulation.
    - diffusive_knudsen : Calculate the diffusive Knudsen number.
    - coulomb_potential_ratio : Compute Coulomb potential ratio.
    - friction_factor : Compute the effective friction factor.

    Examples:
        ```py title="Example Usage"
        import numpy as np
        import particula as par
        kernel_strategy = par.dynamics.HardSphereKernelStrategy()
        charged_coag = par.dynamics.ChargedCoagulationStrategy(
            distribution_type="discrete", kernel_strategy=kernel_strategy
        )
        # Now the charged_coag object can compute dimensionless and
        # dimensioned kernels given a ParticleRepresentation object.
        ```

    References:
        - Seinfeld, J. H., & Pandis, S. N. "Atmospheric Chemistry and
          Physics: From Air Pollution to Climate Change." Wiley, 2016.
    """

    def __init__(
        self,
        distribution_type: str,
        kernel_strategy: ChargedKernelStrategyABC,
    ):
        """Initialize the ChargedCoagulationStrategy.

        Arguments:
            distribution_type : The distribution type representing how
            particles are tracked (e.g., "discrete", "continuous_pdf",
            or "particle_resolved").
            kernel_strategy : A ChargedKernelStrategyABC instance used to
            calculate dimensionless/dimensioned kernels for charged
            coagulation.

        Returns:
            None
        """
        super().__init__(distribution_type=distribution_type)
        self.kernel_strategy = kernel_strategy

    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute the dimensionless kernel for charged coagulation.

        This method delegates computation to the provided kernel strategy. It
        returns the dimensionless kernel (H) as a function of the diffusive
        Knudsen number and the Coulomb potential ratio.

        Arguments:
            - diffusive_knudsen : Dimensionless Knudsen number(s) describing
              particle diffusive behavior.
            - coulomb_potential_ratio : Dimensionless ratio(s) incorporating
              electrostatic interactions.

        Returns:
            - NDArray[np.float64] : Array of dimensionless kernel values.

        Examples:
            ```py title="Dimensionless Kernel Example"
            kn = np.array([0.1, 0.2])
            phi = np.array([1.0, 2.0])
            dim_kernel = charged_coag.dimensionless_kernel(kn, phi)
            # dim_kernel -> array of dimensionless kernel values
            ```
        """
        return self.kernel_strategy.dimensionless(
            diffusive_knudsen=diffusive_knudsen,
            coulomb_potential_ratio=coulomb_potential_ratio,
        )

    def kernel(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Compute the dimensioned coagulation kernel for charged particles.

        This method converts the dimensionless kernel into a dimensioned
        coagulation kernel by combining Coulomb parameters, the pairwise
        radii of particles, reduced mass, and friction factors.

        Arguments:
            - particle : A ParticleRepresentation instance containing
              distribution, density, and concentration data.
            - temperature : Float specifying the system temperature (K).
            - pressure : Float specifying the system pressure (Pa).

        Returns:
            - float or NDArray[np.float64] : The dimensioned coagulation
              kernel value(s).

        Examples:
            ```py title="Dimensioned Kernel Example"
            kernel_matrix = charged_coag.kernel(
                particle=my_particle, temperature=300, pressure=101325
            )
            # kernel_matrix -> 2D array of size (n_particles, n_particles)
            ```

        References:
        - Gopalakrishnan, R. & Hogan, C. J. "Determination of the Transition
          Regime Collision Kernel from Mean First Passage Times." Aerosol
          Science and Technology, 46: 887-899, 2012.
        """
        diffusive_knudsen = self.diffusive_knudsen(
            particle=particle, temperature=temperature, pressure=pressure
        )
        coulomb_potential_ratio = self.coulomb_potential_ratio(
            particle=particle, temperature=temperature
        )
        dimensionless_kernel = self.dimensionless_kernel(
            diffusive_knudsen=diffusive_knudsen,
            coulomb_potential_ratio=coulomb_potential_ratio,
        )
        friction_factor = self.friction_factor(
            particle=particle, temperature=temperature, pressure=pressure
        )
        # Calculate the pairwise sum of radii
        radius = particle.get_radius()
        sum_of_radii = radius[:, np.newaxis] + radius
        # square matrix of mass
        reduced_mass = get_reduced_self_broadcast(particle.get_mass())
        # square matrix of friction factor
        reduced_friction_factor = get_reduced_self_broadcast(friction_factor)

        return self.kernel_strategy.kernel(
            dimensionless_kernel=dimensionless_kernel,
            coulomb_potential_ratio=coulomb_potential_ratio,
            sum_of_radii=sum_of_radii,
            reduced_mass=reduced_mass,
            reduced_friction_factor=reduced_friction_factor,
        )
