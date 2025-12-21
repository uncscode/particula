"""Calculate wall loss rates for particles in various geometries.

Supports neutral spherical and rectangular rate helpers plus charged wall
loss with image-charge enhancement and optional electric-field drift.
"""

from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

from particula.dynamics.properties.wall_loss_coefficient import (
    get_rectangle_wall_loss_coefficient_via_system_state,
    get_spherical_wall_loss_coefficient_via_system_state,
)
from particula.dynamics.wall_loss.wall_loss_strategies import (
    ChargedWallLossStrategy,
)


# pylint: disable=too-many-positional-arguments, too-many-arguments
def get_spherical_wall_loss_rate(
    wall_eddy_diffusivity: float,
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    particle_concentration: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    chamber_radius: float,
) -> Union[float, NDArray[np.float64]]:
    """Calculate the wall loss rate for a spherical chamber.

    The loss rate is computed as ``L = -k c`` where ``k`` is the wall
    loss coefficient [1/s] and ``c`` is the particle concentration
    [#/m³]. The coefficient is obtained from the spherical wall loss
    coefficient computed via
    :func:`get_spherical_wall_loss_coefficient_via_system_state`.

    Args:
        wall_eddy_diffusivity: Wall eddy diffusivity [s⁻¹].
        particle_radius: Particle radius [m].
        particle_density: Particle density [kg/m³].
        particle_concentration: Particle concentration [#/m³].
        temperature: System temperature [K].
        pressure: System pressure [Pa].
        chamber_radius: Radius of the spherical chamber [m].

    Returns:
        Wall loss rate [#/m³·s] as a scalar or array.

    Examples:
        >>> import particula as par
        >>> rate = par.dynamics.wall_loss.get_spherical_wall_loss_rate(
        ...     wall_eddy_diffusivity=1e-3,
        ...     particle_radius=1e-7,
        ...     particle_density=1000.0,
        ...     particle_concentration=1e11,
        ...     temperature=298.0,
        ...     pressure=101325.0,
        ...     chamber_radius=0.5,
        ... )
        >>> rate  # doctest: +SKIP

    References:
        Wikipedia contributors. "Aerosol dynamics." *Wikipedia*.
        https://en.wikipedia.org/wiki/Aerosol.
    """
    # Step 1: Calculate the wall loss coefficient
    loss_coefficient = get_spherical_wall_loss_coefficient_via_system_state(
        wall_eddy_diffusivity=wall_eddy_diffusivity,
        particle_radius=particle_radius,
        particle_density=particle_density,
        temperature=temperature,
        pressure=pressure,
        chamber_radius=chamber_radius,
    )

    # Step 2: Calculate and return the wall loss rate
    return -loss_coefficient * particle_concentration


# pylint: disable=too-many-positional-arguments, too-many-arguments
def get_rectangle_wall_loss_rate(
    wall_eddy_diffusivity: float,
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    particle_concentration: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    chamber_dimensions: Tuple[float, float, float],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the wall loss rate for a rectangular chamber.

    The loss rate is computed as ``L = -k c`` where ``k`` is the wall
    loss coefficient [1/s] and ``c`` is the particle concentration
    [#/m³]. The coefficient is obtained from the rectangular wall loss
    coefficient computed via
    :func:`get_rectangle_wall_loss_coefficient_via_system_state`.

    Args:
        wall_eddy_diffusivity: Wall eddy diffusivity [s⁻¹].
        particle_radius: Particle radius [m].
        particle_density: Particle density [kg/m³].
        particle_concentration: Particle concentration [#/m³].
        temperature: System temperature [K].
        pressure: System pressure [Pa].
        chamber_dimensions: Chamber dimensions ``(length, width, height)`` [m].

    Returns:
        Wall loss rate [#/m³·s] as a scalar or array.

    Examples:
        >>> import particula as par
        >>> loss_rate = par.dynamics.wall_loss.get_rectangle_wall_loss_rate(
        ...     wall_eddy_diffusivity=1e-4,
        ...     particle_radius=5e-8,
        ...     particle_density=1200.0,
        ...     particle_concentration=2e10,
        ...     temperature=300.0,
        ...     pressure=101325.0,
        ...     chamber_dimensions=(1.0, 0.5, 0.5),
        ... )
        >>> loss_rate  # doctest: +SKIP

    References:
        Hinds, W. C. "Aerosol Technology." 2nd ed. John Wiley & Sons, 1999.
    """
    # Step 1: Calculate the wall loss coefficient
    loss_coefficient = get_rectangle_wall_loss_coefficient_via_system_state(
        wall_eddy_diffusivity=wall_eddy_diffusivity,
        particle_radius=particle_radius,
        particle_density=particle_density,
        temperature=temperature,
        pressure=pressure,
        chamber_dimensions=chamber_dimensions,
    )

    # Step 2: Calculate and return the wall loss rate
    return -loss_coefficient * particle_concentration


def get_charged_wall_loss_rate(
    wall_eddy_diffusivity: float,
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    particle_concentration: Union[float, NDArray[np.float64]],
    particle_charge: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    chamber_geometry: str,
    chamber_radius: Union[float, None] = None,
    chamber_dimensions: Union[Tuple[float, float, float], None] = None,
    wall_potential: float = 0.0,
    wall_electric_field: Union[float, Tuple[float, float, float]] = 0.0,
) -> Union[float, NDArray[np.float64]]:
    """Calculate charged wall loss rate for spherical or rectangular chambers.

    Combines neutral deposition with image-charge enhancement and optional
    electric-field drift. Image-charge effects apply when particles carry
    charge even if ``wall_potential`` is zero; drift is applied only when
    ``wall_electric_field`` is non-zero. Reduces to the neutral rate when
    both charge and electric field are zero.

    Args:
        wall_eddy_diffusivity: Wall eddy diffusivity in s⁻¹.
        particle_radius: Particle radius in meters.
        particle_density: Particle density in kg/m³.
        particle_concentration: Particle concentration in #/m³.
        particle_charge: Particle charge in elementary charge units.
        temperature: Gas temperature in kelvin.
        pressure: Gas pressure in pascals.
        chamber_geometry: Geometry name, ``"spherical"`` or
            ``"rectangular"``.
        chamber_radius: Chamber radius in meters for spherical geometry.
        chamber_dimensions: ``(length, width, height)`` in meters for
            rectangular geometry.
        wall_potential: Wall potential in volts. Zero still enables
            image-charge effects when charge is non-zero.
        wall_electric_field: Electric field magnitude in V/m (scalar for
            spherical or tuple for rectangular). Zero disables drift.

    Returns:
        Charged wall loss rate in #/m³·s as scalar or array matching
        ``particle_concentration``.

    Raises:
        ValueError: If geometry arguments are inconsistent or not finite.
    """
    strategy = ChargedWallLossStrategy(
        wall_eddy_diffusivity=wall_eddy_diffusivity,
        chamber_geometry=chamber_geometry,
        chamber_radius=chamber_radius,
        chamber_dimensions=chamber_dimensions,
        wall_potential=wall_potential,
        wall_electric_field=wall_electric_field,
        distribution_type="discrete",
    )
    radius_array = np.asarray(particle_radius, dtype=float)
    density_array = np.asarray(particle_density, dtype=float)
    charge_array = np.asarray(particle_charge, dtype=float)
    coefficient = strategy._combine_coefficients(  # pylint: disable=protected-access
        neutral=strategy._neutral_coefficient(  # pylint: disable=protected-access
            particle_radius=radius_array,
            particle_density=density_array,
            temperature=temperature,
            pressure=pressure,
        ),
        electrostatic_factor=strategy._electrostatic_factor(  # pylint: disable=protected-access
            particle_radius=radius_array,
            particle_charge=charge_array,
            temperature=temperature,
        ),
        drift_term=strategy._drift_term(  # pylint: disable=protected-access
            particle_radius=radius_array,
            particle_charge=charge_array,
            temperature=temperature,
            pressure=pressure,
        ),
    )
    return -coefficient * np.asarray(particle_concentration, dtype=float)
