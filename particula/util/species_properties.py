""" calculate the species properties, saturation vapor pressures,
    latent heats, etc.

    add a more general class, with a dictionary that can be added from
    a file or inputs.
"""
# pylint: disable=all
import numpy as np
from particula import u
from particula.util.input_handling import (in_latent_heat, in_temperature,
                                           in_pressure, in_concentration,
                                           in_scalar
                                           )
from particula.constants import GAS_CONSTANT


class MaterialProperties:
    def __init__(self, species_properties):
        self.species_properties = species_properties

    def __getitem__(self, species):
        return self.species_properties[species]

    def saturation_pressure(self, temperature, species):
        temperature = in_temperature(temperature)
        try:
            if species == 'water':
                return self[species]['saturation_pressure'](temperature)
            else:
                return in_pressure(self[species]['saturation_pressure'])
        except KeyError:
            raise ValueError("Species not implemented")

    def latent_heat(self, temperature, species):
        temperature = in_temperature(temperature)
        try:
            if species == 'water':
                return in_latent_heat(
                        self[species]['latent_heat'](temperature)
                    )
            else:
                return in_latent_heat(self[species]['latent_heat'])
        except KeyError:
            raise ValueError("Species not implemented")


def water_buck_psat(temperature):
    """ Buck equation for water vapor pressure
        https://en.wikipedia.org/wiki/Arden_Buck_equation
    """
    temperature = in_temperature(temperature)
    temp = temperature.m_as("degC")

    return in_pressure(
        6.1115 * np.exp(
            (23.036-temp/333.7)*(temp/(279.82+temp))
        )*u.hPa * (temp < 0.0) + 6.1121 * np.exp(
            (18.678-temp/234.5)*(temp/(257.14+temp))
        )*u.hPa * (temp >= 0.0)
    )


def clausius_clapeyron(
            temperature,
            vapor_pressure,
            temperature_new,
            heat_vaporization
        ):
    """
    Calculates the vapor pressure of a substance at a given temperature
    using the Clausius-Clapeyron equation.

    Parameters
    ----------
        temperature (float): Temperature reference in Kelvin
        vapor_pressure (float): Vapor pressure reference in Pa
        temperature_new (float): Temperature new in Kelvin
        heat_vaporization (float): Heat of vaporization in J/kg

    Returns
    -------
        vapor_pressure_new: Vapor pressure in Pa
    """
    temperature = in_temperature(temperature)
    temperature_new = in_temperature(temperature_new)
    vapor_pressure = in_pressure(vapor_pressure)
    heat_vaporization = in_latent_heat(heat_vaporization)

    return vapor_pressure * np.exp(
        (-heat_vaporization / GAS_CONSTANT)
        * ((1 / temperature_new) - (1 / temperature))
    )


# maybe this should be loaded from a file, or for an option of the user to
# add their own species or load files with species properties
species_properties = {
    "generic": {
        "molecular_weight": 200.0 * u.g / u.mol,
        "surface_tension": 0.072 * u.N / u.m,
        "density": 1000.0 * u.kg / u.m ** 3,
        "vapor_radius": 1e-14 * u.Pa,
        "vapor_attachment": 1.0 * u.dimensionless,
        "saturation_pressure": 1.0 * u.Pa,
        "latent_heat": 1.0 * u.J / u.g,
        "heat_vaporization": 1.0 * u.J / u.kg,
        "kappa": 0.2 * u.dimensionless,
    },

    "water": {
        "molecular_weight": 18.01528 * u.g / u.mol,
        "surface_tension": 0.072 * u.N / u.m,
        "density": 1000.0 * u.kg / u.m ** 3,
        "vapor_radius": 1.6e-9 * u.m,
        "vapor_attachment": 1.0 * u.dimensionless,
        "saturation_pressure": water_buck_psat,
        "latent_heat": lambda T: (
                2500.8 - 2.36*T.m_as('degC')
                + 0.0016*T.m_as('degC')**2
                - 0.00006*T.m_as('degC')**3
            ) * u.J/u.g,
        "heat_vaporization": 2.257e6 * u.J / u.kg,
        "kappa": 0.2 * u.dimensionless,
    },

    "ammonium sulfate": {
        "molecular_weight": 132.14 * u.g / u.mol,
        "surface_tension": 0.072 * u.N / u.m,
        "density": 1770.0 * u.kg / u.m ** 3,
        "vapor_radius": 1.6e-9 * u.m,
        "vapor_attachment": 1.0 * u.dimensionless,
        "saturation_pressure": 1e-14 * u.Pa,
        "latent_heat": 0.0 * u.J/u.g,
        "heat_vaporization": 0.0 * u.J / u.kg,
        "kappa": 0.53 * u.dimensionless,
    }
}


def material_properties(property, species="water", temperature=298.15 * u.K):
    """Return the material properties for a given species.

    Parameters
    ----------
    property : str
        Property to return. Options are: 'all', 'molecular_weight',
        'surface_tension', 'density', 'vapor_radius', 'vapor_attachment',
        'kappa'.
    species : str
        Species for which to return the material properties.
    temperature : K
        Temperature of the material.

    Returns
    -------
    material_properties : value
        The material property for the given species.
    """

    if species in species_properties:
        material_props = MaterialProperties(species_properties)

    if property == 'all':
        return material_props[species]
    elif property in material_props[species]:
        if property == 'saturation_pressure':
            return material_props.saturation_pressure(temperature, species)
        elif property == 'latent_heat':
            return material_props.latent_heat(temperature, species)
        else:
            # no temperature dependence implemented
            return material_props[species][property]
    else:
        raise ValueError("Property not implemented")


def vapor_concentration(
        saturation_ratio,
        temperature=298.15 * u.K,
        species="water"
        ):
    """Convert saturation ratio to mass concentration at a given temperature.

    Parameters
    ----------
    sat_ratio : float
        saturation ratio.
    temperature : K
        Air temperature.
    species : str, optional
        Species for which to calculate the saturation vapor pressure.
        Default is "water".

    Returns
    -------
    concentration : concentration units
        Concentration vapor.

    TODO: check values for different species
    """
    saturation_ratio = in_scalar(saturation_ratio)
    # Calculate saturation vapor pressure
    saturation_pressure = material_properties(
            'saturation_pressure',
            species,
            temperature
        )

    return in_concentration(
        saturation_ratio
        * saturation_pressure
        / (GAS_CONSTANT * temperature)
        * material_properties('molecular_weight', species)
    )
