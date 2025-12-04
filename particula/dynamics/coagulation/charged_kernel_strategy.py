"""Dimensionless coagulation strategies and builders.

This module provides classes for calculating dimensionless (and
optionally dimensioned) coagulation kernels in the transition regime,
including effects of the Coulomb potential between charged particles.

Classes:
    - ChargedKernelStrategyABC : Abstract base class defining the interface
      for dimensionless coagulation strategies.
    - HardSphereKernelStrategy : Hard sphere idealized Coulomb forces.
    - CoulombDyachkov2007KernelStrategy : Dyachkov et al. (2007) approximation.
    - CoulombGatti2008KernelStrategy : Gatti and Kortshagen (2008)
      approximation.
    - CoulombGopalakrishnan2012KernelStrategy : Gopalakrishnan and Hogan (2012)
      approximation.
    - CoulumbChahl2019KernelStrategy : Chahl and Gopalakrishnan (2019)
      approximation.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from particula.dynamics.coagulation import charged_dimensionless_kernel


class ChargedKernelStrategyABC(ABC):
    """Abstract base class for dimensionless coagulation strategies.

    This class defines the dimensionless kernel (H) method, which must be
    implemented by subclasses, and the `kernel` method that converts the
    dimensionless kernel into a dimensioned coagulation kernel.

    Methods:
    - dimensionless : Compute the dimensionless coagulation kernel (H).
    - kernel : Convert a dimensionless kernel into a dimensioned kernel.

    Examples:
        ```py
        class CustomKernel(ChargedKernelStrategyABC):
            def dimensionless(self, diff_kn, phi):
                # user-defined approaches
                return np.ones_like(diff_kn)

        kernel_strategy = CustomKernel()
        dim_kernel = kernel_strategy.kernel(
            dimensionless_kernel=kernel_strategy.dimensionless(...),
            coulomb_potential_ratio=...,
            sum_of_radii=...,
            reduced_mass=...,
            reduced_friction_factor=...
        )
        ```

    References:
        - See references in the individual subclasses for details on specific
          Coulomb approximations.
    """

    @abstractmethod
    def dimensionless(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the dimensionless coagulation kernel (H).

        Arguments:
            - diffusive_knudsen : The diffusive Knudsen number (KₙD)
              [dimensionless].
            - coulomb_potential_ratio : The Coulomb potential ratio (φ_E)
              [dimensionless].

        Returns:
            - NDArray[np.float64] : The dimensionless coagulation kernel (H).

        References:
            - Dyachkov, S. A., et al. (2007).
            - Gatti, M., & Kortshagen, U. (2008).
            - Gopalakrishnan, R., & Hogan, C. J. (2012).
            - Chahl, H. S., & Gopalakrishnan, R. (2019).
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
        """Convert a dimensionless kernel into a dimensioned coagulation kernel.

        Uses reduced mass, friction factors, and particle radii to obtain units
        of [m³/s] for each particle-particle interaction.

        Arguments:
            - dimensionless_kernel : The dimensionless coagulation kernel (H)
              [dimensionless].
            - coulomb_potential_ratio : The Coulomb potential ratio
              [dimensionless].
            - sum_of_radii : The sum of the two particle radii [m].
            - reduced_mass : The reduced mass of the two particles [kg].
            - reduced_friction_factor : The reduced friction factor
              [dimensionless].

        Returns:
            - The dimensioned coagulation kernel [m³/s].

        Examples:
            ```py title="Kernel Conversion Example"
            dim_kernel = kernel_strategy.kernel(
                dimensionless_kernel=H,
                coulomb_potential_ratio=phi,
                sum_of_radii=r_sum,
                reduced_mass=m_reduced,
                reduced_friction_factor=zeta
            )
            ```
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
class HardSphereKernelStrategy(ChargedKernelStrategyABC):
    """Hard sphere dimensionless coagulation strategy.

    Idealized Coulomb interactions and assumes particles interact as
    perfectly charged spheres.

    Methods:
    - dimensionless : Compute the dimensionless kernel under hard sphere
      assumptions.

    Examples:
        ```py title="Hard Sphere Kernel Strategy"
        import particula as par
        hs_strategy = par.dynamics.HardSphereKernelStrategy()
        H = hs_strategy.dimensionless(
            diffusive_knudsen, coulomb_potential_ratio
        )
        # H is the hard-sphere dimensionless kernel
        ```

    References:
    - Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
      in aerosols and dusty plasmas. Phys. Rev. E, 85(2).
      [DOI](https://doi.org/10.1103/PhysRevE.85.026410)
    """

    def dimensionless(  # noqa: D102
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],  # type: ignore
    ) -> NDArray[np.float64]:
        return charged_dimensionless_kernel.get_hard_sphere_kernel(
            diffusive_knudsen  # type: ignore
        )


class CoulombDyachkov2007KernelStrategy(ChargedKernelStrategyABC):
    """Dyachkov et al. (2007) dimensionless coagulation kernel.

    Accounts for Coulomb potential between particles, suitable for
    transition regime calculations.

    Methods:
    - dimensionless : Return the dimensionless kernel (H) following Dyachkov
      et al. (2007).

    Examples:
        ```py title="Use Dyachkov Kernel Strategy"
        import particula as par
        strategy = par.dynamics.CoulombDyachkov2007KernelStrategy()
        H = strategy.dimensionless(diffusive_knudsen, coulomb_potential_ratio)
        ```

    References:
    - Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation
      of particles in the transition regime: The effect of the Coulomb
      potential. J. Chem. Phys., 126(12).
      [DOI](https://doi.org/10.1063/1.2713719)
    """

    def dimensionless(  # noqa: D102
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return charged_dimensionless_kernel.get_coulomb_kernel_dyachkov2007(
            diffusive_knudsen, coulomb_potential_ratio
        )  # type: ignore


class CoulombGatti2008KernelStrategy(ChargedKernelStrategyABC):
    """Gatti & Kortshagen (2008) dimensionless coagulation kernel.

    Captures Coulomb potential effects for a broad range of charge and
    collisionality conditions.

    Methods:
    - dimensionless : Return the dimensionless kernel (H) following Gatti
      and Kortshagen (2008).

    Examples:
        ```py title="Use Gatti Kernel Strategy"
        import particula as par
        strategy = par.dynamics.CoulombGatti2008KernelStrategy()
        H = strategy.dimensionless(diff_kn, phi_ratio)
        ```

    References:
    - Gatti, M., & Kortshagen, U. (2008). Analytical model of particle
      charging in plasmas over a wide range of collisionality. Phys. Rev. E,
      78(4).
      [DOI](https://doi.org/10.1103/PhysRevE.78.046402)
    """

    def dimensionless(  # noqa: D102
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return charged_dimensionless_kernel.get_coulomb_kernel_gatti2008(
            diffusive_knudsen, coulomb_potential_ratio
        )  # type: ignore


class CoulombGopalakrishnan2012KernelStrategy(ChargedKernelStrategyABC):
    """Gopalakrishnan & Hogan (2012) dimensionless coagulation kernel.

    Incorporates Coulomb-influenced collisions in aerosol and dusty plasma
    environments.

    Methods:
    - dimensionless : Return the dimensionless kernel (H) following
      Gopalakrishnan & Hogan (2012).

    Examples:
        ```py title="Use Gopalakrishnan Kernel Strategy"
        import particula as par
        strategy = par.dynamics.CoulombGopalakrishnan2012KernelStrategy()
        H = strategy.dimensionless(kn, phi_ratio)
        ```

    References:
    - Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
      in aerosols and dusty plasmas. Phys. Rev. E, 85(2).
      [DOI](https://doi.org/10.1103/PhysRevE.85.026410)
    """

    def dimensionless(  # noqa: D102
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        result = (
            charged_dimensionless_kernel.get_coulomb_kernel_gopalakrishnan2012(
                diffusive_knudsen, coulomb_potential_ratio
            )
        )
        return np.asarray(result, dtype=np.float64)


class CoulumbChahl2019KernelStrategy(ChargedKernelStrategyABC):
    """Chahl & Gopalakrishnan (2019) dimensionless coagulation kernel.

    Focuses on high-potential, near-free molecular regime Coulombic collisions
    in aerosols and dusty plasmas.

    Methods:
    - dimensionless : Return the dimensionless kernel (H) following
      Chahl & Gopalakrishnan (2019).

    Examples:
        ```py title="Use Chahl 2019 Kernel Strategy"
        import particula as par
        strategy = CoulumbChahl2019KernelStrategy()
        H = strategy.dimensionless(kn, phi)
        ```

    References:
    - Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
      molecular regime Coulombic collisions in aerosols and dusty plasmas.
      Aerosol Sci. Technol., 53(8).
      [DOI](https://doi.org/10.1080/02786826.2019.1614522)
    """

    def dimensionless(  # noqa: D102
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return charged_dimensionless_kernel.get_coulomb_kernel_chahl2019(
            diffusive_knudsen, coulomb_potential_ratio
        )  # type: ignore
