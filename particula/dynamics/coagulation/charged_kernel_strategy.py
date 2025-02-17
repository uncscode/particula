"""
A dimensionless coagulation strategies and builders

This module provides a set of classes and functions to calculate the
dimensionless coagulation kernel for particles in the transition regime.

Classes:
--------
- KernelStrategy: Abstract class for dimensionless coagulation strategies.
- HardSphere: Hard sphere dimensionless coagulation strategy.
- CoulombDyachkov2007: Dyachkov et al. (2007) approximation for the
dimensionless coagulation kernel. Accounts for the Coulomb potential between
particles.
- CoulombGatti2008: Gatti and Kortshagen (2008) approximation for the
dimensionless coagulation kernel. Accounts for the Coulomb potential
between particles.
- CoulombGopalakrishnan2012: Gopalakrishnan and Hogan (2012) approximation for
the dimensionless coagulation kernel. Accounts for the Coulomb potential
between particles.
- CoulumbChahl2019: Chahl and Gopalakrishnan (2019) approximation for the
dimensionless coagulation kernel. Accounts for the Coulomb potential between
particles.
"""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np

from particula.dynamics.coagulation import charged_dimensionless_kernel


class KernelStrategy(ABC):
    """
    Abstract class for dimensionless coagulation strategies. This class defines
    the dimensionless kernel (H) method that must be implemented by any
    dimensionless coagulation strategy.

    Methods:
    --------
    - dimensionless (abstractmethod): Calculate the dimensionless coagulation
    kernel.
    - kernel: Calculate the dimensioned coagulation kernel.
    """

    @abstractmethod
    def dimensionless(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the dimensionless coagulation kernel (H)

        Args:
        -----
        - diffusive_knudsen: The diffusive Knudsen number (K_nD)
        [dimensionless].
        - coulomb_potential_ratio: The Coulomb potential ratio (phi_E)
        [dimensionless].

        Returns:
        --------
        The dimensionless coagulation kernel (H) [dimensionless].

        References:
        -----------
        - Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation
        of particles in the transition regime: The effect of the Coulomb
        potential. Journal of Chemical Physics, 126(12).
        https://doi.org/10.1063/1.2713719
        - Gatti, M., & Kortshagen, U. (2008). Analytical model of particle
        charging in plasmas over a wide range of collisionality. Physical
        Review E - Statistical, Nonlinear, and Soft Matter Physics, 78(4).
        https://doi.org/10.1103/PhysRevE.78.046402
        - Gopalakrishnan, R., & Hogan, C. J. (2011). Determination of the
        transition regime collision kernel from mean first passage times.
        Aerosol Science and Technology, 45(12), 1499-1509.
        https://doi.org/10.1080/02786826.2011.601775
        - Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced
        collisions in aerosols and dusty plasmas. Physical Review E -
        Statistical, Nonlinear, and Soft Matter Physics, 85(2).
        https://doi.org/10.1103/PhysRevE.85.026410
        - Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
        molecular regime Coulombic collisions in aerosols and dusty plasmas.
        Aerosol Science and Technology, 53(8), 933-957.
        https://doi.org/10.1080/02786826.2019.1614522
        """

    def kernel(
        self,
        dimensionless_kernel: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
        sum_of_radii: NDArray[np.float64],
        reduced_mass: NDArray[np.float64],
        reduced_friction_factor: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # pylint: disable=too-many-positional-arguments, too-many-arguments
        """
        The dimensioned coagulation kernel for each particle pair, calculated
        from the dimensionless coagulation kernel and the reduced quantities.
        All inputs are square matrices, for all particle-particle interactions.

        Arguments:
            - dimensionless_kernel : The dimensionless coagulation kernel
                [dimensionless].
            - coulomb_potential_ratio : The Coulomb potential ratio
                [dimensionless].
            - sum_of_radii : The sum of the radii of the particles [m].
            - reduced_mass : The reduced mass of the particles [kg].
            - reduced_friction_factor : The reduced friction factor of the
                particles [dimensionless].

        Returns:
            - The dimensioned coagulation kernel, as a square matrix, of all
                particle-particle interactions [m^3/s].

        *References*:
        - Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
        molecular regime Coulombic collisions in aerosols and dusty plasmas.
        Aerosol Science and Technology, 53(8), 933-957.
        https://doi.org/10.1080/02786826.2019.1614522
        """
        # pylint: disable=duplicate-code
        return charged_dimensionless_kernel.get_dimensional_kernel(
            dimensionless_kernel=dimensionless_kernel,
            coulomb_potential_ratio=coulomb_potential_ratio,
            sum_of_radii=sum_of_radii,
            reduced_mass=reduced_mass,
            reduced_friction_factor=reduced_friction_factor,
        )


# define strategies
class HardSphereKernelStrategy(KernelStrategy):
    """
    Hard sphere dimensionless coagulation strategy.
    """

    def dimensionless(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],  # type: ignore
    ) -> NDArray[np.float64]:
        return charged_dimensionless_kernel.get_hard_sphere_kernel(
            diffusive_knudsen
        )


class CoulombDyachkov2007KernelStrategy(KernelStrategy):
    """
    Dyachkov et al. (2007) approximation for the dimensionless coagulation
    kernel. Accounts for the Coulomb potential between particles.

    References:
    - Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
    particles in the transition regime: The effect of the Coulomb potential.
    Journal of Chemical Physics, 126(12).
    https://doi.org/10.1063/1.2713719
    """

    def dimensionless(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return charged_dimensionless_kernel.get_coulomb_kernel_dyachkov2007(
            diffusive_knudsen, coulomb_potential_ratio
        )  # type: ignore


class CoulombGatti2008KernelStrategy(KernelStrategy):
    """
    Gatti and Kortshagen (2008) approximation for the dimensionless coagulation
    kernel. Accounts for the Coulomb potential between particles.

    References:
    -----------
    - Gatti, M., & Kortshagen, U. (2008). Analytical model of particle
    charging in plasmas over a wide range of collisionality. Physical Review
    E - Statistical, Nonlinear, and Soft Matter Physics, 78(4).
    https://doi.org/10.1103/PhysRevE.78.046402
    """

    def dimensionless(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return charged_dimensionless_kernel.get_coulomb_kernel_gatti2008(
            diffusive_knudsen, coulomb_potential_ratio
        )  # type: ignore


class CoulombGopalakrishnan2012KernelStrategy(KernelStrategy):
    """
    Gopalakrishnan and Hogan (2012) approximation for the dimensionless
    coagulation kernel. Accounts for the Coulomb potential between particles.

    References:
    -----------
    - Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
    in aerosols and dusty plasmas. Physical Review E - Statistical, Nonlinear,
    and Soft Matter Physics, 85(2).
    https://doi.org/10.1103/PhysRevE.85.026410
    """

    def dimensionless(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return (
            charged_dimensionless_kernel.get_coulomb_kernel_gopalakrishnan2012(
                diffusive_knudsen, coulomb_potential_ratio
            )
        )


class CoulumbChahl2019KernelStrategy(KernelStrategy):
    """
    Chahl and Gopalakrishnan (2019) approximation for the dimensionless
    coagulation kernel. Accounts for the Coulomb potential between particles.

    References:
    -----------
    - Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
    molecular regime Coulombic collisions in aerosols and dusty plasmas.
    Aerosol Science and Technology, 53(8), 933-957.
    https://doi.org/10.1080/02786826.2019.1614522
    """

    def dimensionless(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return charged_dimensionless_kernel.get_coulomb_kernel_chahl2019(
            diffusive_knudsen, coulomb_potential_ratio
        )
