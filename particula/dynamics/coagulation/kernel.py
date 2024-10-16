""" A dimensionless coagulation strategies and builders
"""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np

from particula.particles.properties import coulomb_enhancement
from particula.dynamics.coagulation import transition_regime


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

    def kernel(  # pylint: disable=too-many-positional-arguments, too-many-arguments
        self,
        dimensionless_kernel: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
        sum_of_radii: NDArray[np.float64],
        reduced_mass: NDArray[np.float64],
        reduced_friction_factor: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        The dimensioned coagulation kernel for each particle pair, calculated
        from the dimensionless coagulation kernel and the reduced quantities.
        All inputs are square matrices, for all particle-particle interactions.

        Args:
        -----
        - dimensionless_kernel: The dimensionless coagulation kernel
        [dimensionless].
        - coulomb_potential_ratio: The Coulomb potential ratio [dimensionless].
        - sum_of_radii: The sum of the radii of the particles [m].
        - reduced_mass: The reduced mass of the particles [kg].
        - reduced_friction_factor: The reduced friction factor of the
        particles [dimensionless].

        Returns:
        --------
        The dimensioned coagulation kernel, as a square matrix, of all
        particle-particle interactions [m^3/s].

        Check, were the /s comes from.

        References:
        -----------
        """
        coulomb_kinetic_limit = coulomb_enhancement.kinetic(
            coulomb_potential_ratio
        )
        coulomb_continuum_limit = coulomb_enhancement.continuum(
            coulomb_potential_ratio
        )
        return (
            dimensionless_kernel
            * reduced_friction_factor
            * sum_of_radii**3
            * coulomb_kinetic_limit**2
            / (reduced_mass * coulomb_continuum_limit)
        )


# define strategies
class HardSphere(KernelStrategy):
    """
    Hard sphere dimensionless coagulation strategy.
    """

    def dimensionless(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],  # type: ignore
    ) -> NDArray[np.float64]:
        return transition_regime.hard_sphere(diffusive_knudsen)  # type: ignore


class CoulombDyachkov2007(KernelStrategy):
    """
    Dyachkov et al. (2007) approximation for the dimensionless coagulation
    kernel. Accounts for the Coulomb potential between particles.

    References:
    -----------
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
        return transition_regime.coulomb_dyachkov2007(
            diffusive_knudsen, coulomb_potential_ratio
        )  # type: ignore


class CoulombGatti2008(KernelStrategy):
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
        return transition_regime.coulomb_gatti2008(
            diffusive_knudsen, coulomb_potential_ratio
        )  # type: ignore


class CoulombGopalakrishnan2012(KernelStrategy):
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
        return transition_regime.coulomb_gopalakrishnan2012(
            diffusive_knudsen, coulomb_potential_ratio
        )  # type: ignore


class CoulumbChahl2019(KernelStrategy):
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
        return transition_regime.coulomb_chahl2019(
            diffusive_knudsen, coulomb_potential_ratio
        )  # type: ignore
