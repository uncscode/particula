""" handling inputs
"""

from particula import u


def in_radius(radius):
    """ Handles radius input

        Parameters:
            radius    (float) [m | dimensionless]

        Returns:
            radius    (float) [m]

        Notes:
            * If unit is correct, take to base units in m
            * Throws ValueError if unit is wrong
            * Assigning m units to scalar input
    """

    if radius is None:
        raise ValueError("You must provide a radius!")

    if isinstance(radius, u.Quantity):
        if radius.to_base_units().u == u.m:
            radius = radius.to_base_units()
        else:
            raise ValueError(
                f"\n\t"
                f"Input {radius} has unsupported units.\n\t"
                f"Input must have radius units of\n\t"
                f"'meters';\n\t"
                f"otherwise, if dimensionless, it will\n\t"
                f"be assigned a m unit.\n"
            )
    else:
        radius = u.Quantity(radius, u.m)

    return radius


def in_handling(value, units: u.Quantity):
    """ generic function to handle inputs

        Parameters:
            value     (float)       [u.Quantity | dimensionless]
            units     (u.Quantity)

        Returns:
            value     (float)       [u.Quantity]

        Notes:
            * If unit is correct, take to base units
            * Throws ValueError if unit is wrong
            * Assigning default base units to scalar input
    """

    if isinstance(value, u.Quantity):
        if value.to_base_units().u == (1*units).to_base_units().u:
            value = value.to_base_units()
        else:
            raise ValueError(
                f"\n\t"
                f"Input has unsupported units.\n\t"
                f"Input must have units equivlanet to {units};\n\t"
                f"otherwise, if dimensionless, it will\n\t"
                f"be assigned {units}.\n"
            )
    else:
        value = u.Quantity(value, (1*units).to_base_units().u)

    return value


# pylint: disable=missing-docstring, multiple-statements
def in_temperature(temp): return in_handling(temp, u.K)
def in_viscosity(vis): return in_handling(vis, u.kg/u.m/u.s)
def in_pressure(pres): return in_handling(pres, u.Pa)
def in_mass(mass): return in_handling(mass, u.kg)
def in_volume(vol): return in_handling(vol, u.m**3)
def in_time(time): return in_handling(time, u.s)
def in_velocity(vel): return in_handling(vel, u.m/u.s)
def in_acceleration(acc): return in_handling(acc, u.m/u.s**2)
def in_molecular_weight(molw): return in_handling(molw, u.kg/u.mol)
def in_density(density): return in_handling(density, u.kg/u.m**3)
def in_scalar(scalar): return in_handling(scalar, u.dimensionless)
def in_length(length): return in_handling(length, u.m)
def in_area(area): return in_handling(area, u.m**2)
def in_gas_constant(con): return in_handling(con, u.J/u.K/u.mol)
def in_concentration(conc): return in_handling(conc, u.kg/u.m**3)
