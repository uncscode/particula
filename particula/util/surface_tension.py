""" calculating the surface tension of a mixture of solutes and water """
# pylint: disable=too-many-arguments

import numpy as np
from particula.util.input_handling import in_surface_tension, in_temperature, \
    in_scalar, in_volume, in_radius


def water(
        temperature,
        critical_temperature=647.15):
    """Calculate the surface tension of water using the equation from Kalova
    and Mares (2018).

    Args:
    ----------
        Temperature : float, Ambient temperature of air
        CritTemp : float, optional: Critical temperature of water

    Returns:
    -------
        sigma : float, Surface tension of water at the given temperature
    """
    temperature = in_temperature(temperature)
    critical_temperature = in_temperature(critical_temperature)
    # Dimensionless parameter from fitting equation
    tau = 1 - temperature / critical_temperature

    # Surface tension in mN/m
    sigma = 241.322 * (tau.m ** 1.26) * (
        1 - 0.0589 * (tau.m ** 0.5) - 0.56917 * tau.m
    )

    return in_surface_tension(sigma / 1000)


def dry_mixing(volume_fractions, surface_tensions):
    """Function to calculate the effective surface tension of a dry mixture.

    Args:
    ----------
    volume_fractions : array, volume fractions of solutes
    surface_tensions : array, surface tensions of solutes

    Returns:
    --------
    sigma : array, surface tension of droplet
    """
    volume_fractions = in_scalar(volume_fractions)
    surface_tensions = in_surface_tension(surface_tensions)
    if np.sum(volume_fractions) != 1:
        volume_fractions = volume_fractions / np.sum(volume_fractions)

    # Calculate the surface tension of the mixture
    return np.sum(volume_fractions * surface_tensions)


def wet_mixing(
    volume_solute,
    volume_water,
    wet_radius,
    surface_tension_solute,
    temperature,
    method='film'
):
    """Function to calculate the effective surface tension of a wet mixture.

    Args:
    ----------
    volume_solute : array, volume of solute mixture
    volume_water : array, volume of water
    surface_tension_solute : array, surface tension of solute mixture
    temperature : float, temperature of droplet
    method : str, optional: [film, volume] method to calculate effective
        surface tension

    Returns:
    --------
    EffSigma : array, effective surface tension of droplet
    """

    volume_solute = in_volume(volume_solute)
    volume_water = in_volume(volume_water)
    wet_radius = in_radius(wet_radius)
    surface_tension_solute = in_surface_tension(surface_tension_solute)
    temperature = in_temperature(temperature)

    # Organic-film method
    if method == 'film':

        # Characteristic monolayer thickness taken from AIOMFAC
        mono = in_radius(0.3e-9)

        # Volume of the monolayer
        film_volume = (4 / 3) * np.pi * (wet_radius **
                                         3 - (wet_radius - mono)**3)

        # Determine if there's enough organics to cover the particle
        surface_coverage = volume_solute / film_volume

        if surface_coverage >= 1:
            # If the particle is completely covered, the surface tension is
            # just the surface tension of the solute
            sigma = surface_tension_solute
        else:
            # If the particle is not completely covered, the surface tension
            # is a weighted average of the surface tension of the solute and
            # water
            sigma = surface_tension_solute * surface_coverage + \
                water(temperature) * (1 - surface_coverage)
    elif method == 'volume':
        # Volume method
        sigma = (
            volume_solute * surface_tension_solute +
            volume_water * water(temperature)
        ) / (volume_solute + volume_water)

    return sigma
