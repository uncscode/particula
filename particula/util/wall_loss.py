""" calculate the wall loss coefficient
"""
import numpy as np

from particula import u
from particula.util.input_handling import in_handling
from particula.util.debye_function import df1
from particula.util.settling_velocity import psv
from particula.util.diffusion_coefficient import pdc


def spherical_wall_loss_coefficient(
    ktp_value,
    diffusion_coefficient_value,
    settling_velocity_value,
    chamber_radius
):
    """Calculate the wall loss coefficient for a spherical chamber
        approximation.

    Args:
        ktp_value: rate of the wall eddy diffusivity
        diffusion_coefficient_value: Particle diffusion coefficient.
        settling_velocity_value: Settling velocity of the particle.
        chamber_radius: Radius of the chamber.

    Returns:
        The calculated wall loss coefficient for simple case.
    """
    return (
        6 * np.sqrt(ktp_value * diffusion_coefficient_value) /
        (np.pi * chamber_radius) * df1(
            np.pi * settling_velocity_value /
            (2 * np.sqrt(ktp_value * diffusion_coefficient_value))
        ) + settling_velocity_value / (4 * chamber_radius / 3)
    )


def rectangle_wall_loss(
    ktp_value,
    diffusion_coefficient_value,
    settling_velocity_value,
    dimension
):
    """
    Calculate the wall loss coefficient, β₀, for a rectangular chamber.

    Given the rate of wall diffusivity parameter (ktp_value), the particle
    diffusion coefficient (diffusion_coefficient_value), and the terminal
    settling velocity (settling_velocity_value), this function computes the
    wall loss coefficient for a rectangular-prism chamber with specified
    dimensions.

    The wall loss coefficient is calculated based on the diffusion and
    gravitational sedimentation in a rectangular chamber. It accounts for the
    effect of chamber geometry on particle loss by considering the length (L),
    width (W), and height (H) of the chamber.

    Args:
        ktp_value (float): Rate of wall diffusivity parameter in units of
            inverse seconds (s^-1).
        diffusion_coefficient_value (float): The particle diffusion
            coefficient in units of square meters per second (m^2/s).
        settling_velocity_value (float): The terminal settling velocity of the
            particles, in units of meters per second (m/s).
        dimension (tuple): A tuple of three floats representing the length (L)
            width (W), and height (H) of the rectangular chamber,
            in units of meters (m).

    Returns:
        float: The calculated wall loss coefficient (B0) for the rectangular
        chamber.

    Reference:
        The wall loss coefficient, β₀, is calculated using the following
        formula:
        $$
        \beta_0 = (LWH)^{-1} (4H(L+W) \\sqrt{k_t D}/\\pi +
        v_g LW \\coth{[(\\pi v_g)/(4\\sqrt{k_t D}})])
        $$
    """
    l, w, h = dimension  # Unpack the dimensions tuple
    # Using 1/tanh(x) for coth(x)
    coth_vg_kt_d = 1 / np.tanh(
        (np.pi * settling_velocity_value)
        / (4 * np.sqrt(ktp_value * diffusion_coefficient_value))
        )
    return (l * w * h)**-1 * (
        4 * h * (l + w) * np.sqrt(ktp_value * diffusion_coefficient_value)
        / np.pi
        + settling_velocity_value * l * w * coth_vg_kt_d)


def wlc(
    approx="none",
    ktp_value=0.1 * u.s**-1,
    diffusion_coefficient_value=None,
    dimension=1 * u.m,
    settling_velocity_value=None,
    **kwargs
):
    """Calculate the wall loss coefficient.

    Args:
        approximation: The approximation method to use, e.g., "none",
        "spherical", "rectangle"
        ktp_value: rate of the wall eddy diffusivity
        diffusion_coefficient_value: Particle diffusion coefficient.
        settling_velocity_value: Settling velocity of the particle.
        dimension: Radius of the chamber or tuple of rectangular dimensions.

    Returns:
        The calculated wall loss coefficient.
    """

    if approx == "none":
        return 0.0

    if approx in ["simple", "spherical"]:
        # input checks
        ktp_value = in_handling(ktp_value, u.s**-1)
        diffusion_coefficient_value = in_handling(
            diffusion_coefficient_value, u.m**2 / u.s
        ) if diffusion_coefficient_value is not None else pdc(**kwargs)

        settling_velocity_value = in_handling(
            settling_velocity_value, u.m / u.s
        ) if settling_velocity_value is not None else psv(**kwargs)

        chamber_radius = in_handling(
            dimension, u.m
        ) if dimension is not None else 1 * u.m

        # calculation of the wall loss coefficient
        return spherical_wall_loss_coefficient(
            ktp_value=ktp_value,
            diffusion_coefficient_value=diffusion_coefficient_value,
            settling_velocity_value=settling_velocity_value,
            chamber_radius=chamber_radius
        )

    if approx == "rectangle":
        if dimension is None:
            # Default dimensions if not provided
            dimension = (1 * u.m, 1 * u.m, 1 * u.m)
        else:
            dimension = tuple(in_handling(dim, u.m) for dim in dimension)

        return rectangle_wall_loss(
            ktp_value=ktp_value,
            diffusion_coefficient_value=diffusion_coefficient_value,
            settling_velocity_value=settling_velocity_value,
            dimension=dimension
        )

    return 0
