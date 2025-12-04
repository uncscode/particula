"""Charged dimensional kernel for coagulation calculated from system state."""

# pylint: disable=duplicate-code

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.dynamics.coagulation.charged_dimensionless_kernel import (
    get_coulomb_kernel_chahl2019,
    get_coulomb_kernel_dyachkov2007,
    get_coulomb_kernel_gatti2008,
    get_coulomb_kernel_gopalakrishnan2012,
    get_dimensional_kernel,
    get_hard_sphere_kernel,
)
from particula.gas import (
    get_dynamic_viscosity,
    get_molecule_mean_free_path,
)
from particula.particles import (
    get_coulomb_enhancement_ratio,
    get_cunningham_slip_correction,
    get_diffusive_knudsen_number,
    get_friction_factor,
    get_knudsen_number,
)
from particula.util import get_reduced_self_broadcast


def _system_state_properties(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_mass: Union[float, NDArray[np.float64]],
    particle_charge: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Get the system state properties for charged particles.

    Arguments:
        - particle_radius : The radius of the particles [m].
        - particle_mass : The mass of the particles [kg].
        - particle_charge : The charge of the particles [C].
        - temperature : The temperature of the system [K].
        - pressure : The pressure of the system [Pa].

    Returns:
        - coulomb_potential_ratio : The Coulomb potential ratio
            [dimensionless].
        - diffusive_knudsen : The diffusive knudsen number [dimensionless].
        - sum_of_radii : The sum of the radii of the particles [m].
        - reduced_mass : The reduced mass of the particles [kg].
        - reduced_friction_factor : The reduced friction factor of the
            particles [dimensionless].

    Examples: (This is an internal helper and typically not called directly.)
    """
    # get properties
    dynamic_viscosity = get_dynamic_viscosity(temperature=temperature)

    # get knudsen number
    knudsen_number = get_knudsen_number(
        mean_free_path=get_molecule_mean_free_path(
            temperature=temperature,
            dynamic_viscosity=dynamic_viscosity,
            pressure=pressure,
        ),
        particle_radius=particle_radius,
    )

    # get friction factor
    friction_factor = get_friction_factor(
        particle_radius=particle_radius,
        dynamic_viscosity=get_dynamic_viscosity(temperature=temperature),
        slip_correction=get_cunningham_slip_correction(
            knudsen_number=knudsen_number
        ),
    )

    # get coulomb potential ratio
    coulomb_potential_ratio = get_coulomb_enhancement_ratio(
        particle_radius=particle_radius,
        charge=particle_charge,  # type: ignore
        temperature=temperature,
    )

    # get diffusive knudsen number
    diffusive_knudsen = get_diffusive_knudsen_number(
        particle_radius=particle_radius,
        particle_mass=particle_mass,
        friction_factor=friction_factor,
        coulomb_potential_ratio=coulomb_potential_ratio,
        temperature=temperature,
    )

    sum_of_radii = (
        particle_radius[:, np.newaxis] + particle_radius[np.newaxis, :]
    )
    reduced_mass = get_reduced_self_broadcast(particle_mass)
    reduced_friction_factor = get_reduced_self_broadcast(friction_factor)

    return (
        coulomb_potential_ratio,
        diffusive_knudsen,
        sum_of_radii,
        reduced_mass,
        reduced_friction_factor,
    )


def get_hard_sphere_kernel_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_mass: Union[float, NDArray[np.float64]],
    particle_charge: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]:
    """The hard sphere dimensioned coagulation kernel via system state.

    For the hard sphere kernel, the dimensionless kernel is computed
    internally based on the diffusive Knudsen number, then converted
    to the dimensioned form using the system state properties
    (particle radius, mass, charge, temperature, pressure).
    coulomb potential ratio, sum of radii, reduced mass...etc are all
    calculated from the system state properties (temperature, pressure, etc.).
    These are used to calculate the dimensionless kernel, which is then
    converted to the dimensioned kernel.

    Arguments:
        - particle_radius : The radius of the particles [m].
        - particle_mass : The mass of the particles [kg].
        - particle_charge : The charge of the particles [C].
        - temperature : The temperature of the system [K].
        - pressure : The pressure of the system [Pa].

    Returns:
        - The dimensioned coagulation kernel, as a square matrix, of all
            particle-particle interactions [m^3/s].

    Examples:
        ``` py title="Example Usage"
        kernel_matrix = get_hard_sphere_kernel_via_system_state(
            p_radius, p_mass, p_charge, 298.15, 101325
        )
        # kernel_matrix is an N×N array of rates
        ```

    References:
    - Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
      particles in the transition regime: The effect of the Coulomb potential.
      Journal of Chemical Physics, 126(12).
      https://doi.org/10.1063/1.2713719
    """
    # get system state properties
    (
        coulomb_potential_ratio,
        diffusive_knudsen,
        sum_of_radii,
        reduced_mass,
        reduced_friction_factor,
    ) = _system_state_properties(
        particle_radius=particle_radius,
        particle_mass=particle_mass,
        particle_charge=particle_charge,
        temperature=temperature,
        pressure=pressure,
    )

    # get dimensionless kernel
    dimensionless_kernel = get_hard_sphere_kernel(
        diffusive_knudsen=diffusive_knudsen,
    )

    # get dimensioned kernel
    return get_dimensional_kernel(
        dimensionless_kernel=dimensionless_kernel,
        coulomb_potential_ratio=coulomb_potential_ratio,
        sum_of_radii=sum_of_radii,
        reduced_mass=reduced_mass,
        reduced_friction_factor=reduced_friction_factor,
    )


def get_coulomb_kernel_dyachkov2007_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_mass: Union[float, NDArray[np.float64]],
    particle_charge: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]:
    """The dimensioned coagulation kernel via Dyachkov (2007).

    Arguments:
        - particle_radius : The radius of the particles [m].
        - particle_mass : The mass of the particles [kg].
        - particle_charge : The charge of the particles [C].
        - temperature : The temperature of the system [K].
        - pressure : The pressure of the system [Pa].

    Returns:
        - The dimensioned coagulation kernel, as a square matrix,
          [m^3/s].

    Examples:
        ``` py title="Example Usage"
        import particula as par
        kernel_dyachkov = (
            par.dynamics.get_coulomb_kernel_dyachkov2007_via_system_state(
                p_radius, p_mass, p_charge, 298.15, 101325
            )
        )
        print(kernel_dyachkov.shape)
        ```

    References:
    - Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
      particles in the transition regime: The effect of the Coulomb potential.
      Journal of Chemical Physics, 126(12).
      https://doi.org/10.1063/1.2713719
    """
    (
        coulomb_potential_ratio,
        diffusive_knudsen,
        sum_of_radii,
        reduced_mass,
        reduced_friction_factor,
    ) = _system_state_properties(
        particle_radius=particle_radius,
        particle_mass=particle_mass,
        particle_charge=particle_charge,
        temperature=temperature,
        pressure=pressure,
    )
    dimensionless_kernel = get_coulomb_kernel_dyachkov2007(
        diffusive_knudsen=diffusive_knudsen,
        coulomb_potential_ratio=coulomb_potential_ratio,
    )
    return get_dimensional_kernel(
        dimensionless_kernel=dimensionless_kernel,
        coulomb_potential_ratio=coulomb_potential_ratio,
        sum_of_radii=sum_of_radii,
        reduced_mass=reduced_mass,
        reduced_friction_factor=reduced_friction_factor,
    )


def get_coulomb_kernel_gatti2008_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_mass: Union[float, NDArray[np.float64]],
    particle_charge: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]:
    """The dimensioned coagulation kernel via system state using Gatti (2008).

    Arguments:
        - particle_radius : The radius of the particles [m].
        - particle_mass : The mass of the particles [kg].
        - particle_charge : The charge of the particles [C].
        - temperature : The temperature of the system [K].
        - pressure : The pressure of the system [Pa].

    Returns:
        - The dimensioned coagulation kernel, as a square matrix,
          [m^3/s].

    Examples:
        ``` py title="Example Usage"
        import particula as par
        kernel_gatti = (
            par.dynamics.get_coulomb_kernel_gatti2008_via_system_state(
                p_radius, p_mass, p_charge, 298.15, 101325
            )
        )
        print(kernel_gatti)
        ```

    References:
    - Gatti, M., & Kortshagen, U. (2008). Analytical model of particle
      charging in plasmas over a wide range of collisionality. Physical Review
      E - Statistical, Nonlinear, and Soft Matter Physics, 78(4).
      https://doi.org/10.1103/PhysRevE.78.046402
    """
    (
        coulomb_potential_ratio,
        diffusive_knudsen,
        sum_of_radii,
        reduced_mass,
        reduced_friction_factor,
    ) = _system_state_properties(
        particle_radius=particle_radius,
        particle_mass=particle_mass,
        particle_charge=particle_charge,
        temperature=temperature,
        pressure=pressure,
    )
    dimensionless_kernel = get_coulomb_kernel_gatti2008(
        diffusive_knudsen=diffusive_knudsen,
        coulomb_potential_ratio=coulomb_potential_ratio,
    )
    return get_dimensional_kernel(
        dimensionless_kernel=dimensionless_kernel,
        coulomb_potential_ratio=coulomb_potential_ratio,
        sum_of_radii=sum_of_radii,
        reduced_mass=reduced_mass,
        reduced_friction_factor=reduced_friction_factor,
    )


def get_coulomb_kernel_gopalakrishnan2012_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_mass: Union[float, NDArray[np.float64]],
    particle_charge: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]:
    """Gopalakrishnan (2012) dimensioned coagulation kernel via system state.

    Arguments:
        - particle_radius : The radius of the particles [m].
        - particle_mass : The mass of the particles [kg].
        - particle_charge : The charge of the particles [C].
        - temperature : The temperature of the system [K].
        - pressure : The pressure of the system [Pa].

    Returns:
        - The dimensioned coagulation kernel, as a square matrix,
          [m^3/s].

    Examples:
        ``` py title="Example Usage"
        import particula as par
        kernel_gopal = (
            par.dyanmics.get_coulomb_kernel_gopalakrishnan2012_via_system_state(
                p_radius, p_mass, p_charge, 298.15, 101325
            )
        )
        # kernel_gopal is an N×N matrix [m^3/s]
        ```

    References:
    - Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
      in aerosols and dusty plasmas. Physical Review E - Statistical, Nonlinear
      and Soft Matter Physics, 85(2).
      https://doi.org/10.1103/PhysRevE.85.026410
    """
    (
        coulomb_potential_ratio,
        diffusive_knudsen,
        sum_of_radii,
        reduced_mass,
        reduced_friction_factor,
    ) = _system_state_properties(
        particle_radius=particle_radius,
        particle_mass=particle_mass,
        particle_charge=particle_charge,
        temperature=temperature,
        pressure=pressure,
    )
    dimensionless_kernel = get_coulomb_kernel_gopalakrishnan2012(
        diffusive_knudsen=diffusive_knudsen,
        coulomb_potential_ratio=coulomb_potential_ratio,
    )
    return get_dimensional_kernel(
        dimensionless_kernel=dimensionless_kernel,
        coulomb_potential_ratio=coulomb_potential_ratio,
        sum_of_radii=sum_of_radii,
        reduced_mass=reduced_mass,
        reduced_friction_factor=reduced_friction_factor,
    )


def get_coulomb_kernel_chahl2019_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_mass: Union[float, NDArray[np.float64]],
    particle_charge: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]:
    """Chahl (2019) dimensioned coagulation kernel via system state.

    Arguments:
        - particle_radius : The radius of the particles [m].
        - particle_mass : The mass of the particles [kg].
        - particle_charge : The charge of the particles [C].
        - temperature : The temperature of the system [K].
        - pressure : The pressure of the system [Pa].

    Returns:
        - The dimensioned coagulation kernel, as a square matrix,
          [m^3/s].

    Examples:
        ``` py title="Example Usage"
        import particula as par
        kernel_chahl = ()
            par.dynamics.get_coulomb_kernel_chahl2019_via_system_state(
            p_radius, p_mass, p_charge, 298.15, 101325
            )
        )
        print(kernel_chahl)
        ```

    References:
    - Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
        molecular regime Coulombic collisions in aerosols and dusty plasmas.
        Aerosol Science and Technology, 53(8), 933-957.
        https://doi.org/10.1080/02786826.2019.1614522
    """
    (
        coulomb_potential_ratio,
        diffusive_knudsen,
        sum_of_radii,
        reduced_mass,
        reduced_friction_factor,
    ) = _system_state_properties(
        particle_radius=particle_radius,
        particle_mass=particle_mass,
        particle_charge=particle_charge,
        temperature=temperature,
        pressure=pressure,
    )
    dimensionless_kernel = get_coulomb_kernel_chahl2019(
        diffusive_knudsen=diffusive_knudsen,
        coulomb_potential_ratio=coulomb_potential_ratio,
    )
    return get_dimensional_kernel(
        dimensionless_kernel=dimensionless_kernel,
        coulomb_potential_ratio=coulomb_potential_ratio,
        sum_of_radii=sum_of_radii,
        reduced_mass=reduced_mass,
        reduced_friction_factor=reduced_friction_factor,
    )
