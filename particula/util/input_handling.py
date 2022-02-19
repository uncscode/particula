""" handling inputs
"""

from particula import u

def in_temperature(temp):
    """ Handles temperature input

        Parameters:
            temp    (float) [K | dimensionless]

        Returns:
            temp    (float) [K]

        Notes:
            * If unit is correct, take to base units in kelvin
            * Throws ValueError if unit is wrong
            * Assigning kelvin units to scalar input
    """

    if isinstance(temp, u.Quantity):
        if temp.to_base_units().u == u.kelvin:
            temp = temp.to_base_units()
        else:
            raise ValueError(
                f"\n\t"
                f"Input {temp} has unsupported units.\n\t"
                f"Input must have temperature units of\n\t"
                f"either 'kelvin' or equivalent;\n\t"
                f"otherwise, if dimensionless, it will\n\t"
                f"be assigned a kelvin unit.\n"
            )
    else:
        temp = u.Quantity(temp, u.K)

    return temp

def in_viscosity(vis):
    """ Handles viscosity input

        Parameters:
            vis     (float) [Pa*s | dimensionless]

        Returns:
            vis     (float) [Pa*s]

        Notes:
            * If unit is correct, take to base units in Pa*s
            * Throws ValueError if unit is wrong
            * Assigning Pa*s units to scalar input
    """

    if isinstance(vis, u.Quantity):
        if vis.to_base_units().u == u.kg/u.m/u.s:
            vis = vis.to_base_units()
        else:
            raise ValueError(
                f"\n\t"
                f"Input {vis} has unsupported units.\n\t"
                f"Input must have viscosity units of\n\t"
                f"either 'Pa*s' or equivalent;\n\t"
                f"otherwise, if dimensionless, it will\n\t"
                f"be assigned a Pa*s unit.\n"
            )
    else:
        vis = u.Quantity(vis, u.kg/u.m/u.s)

    return vis

def in_pressure(pressure):
    """ Handles pressure input

        Parameters:
            pressure    (float) [Pa | dimensionless]

        Returns:
            pressure    (float) [Pa]

        Notes:
            * If unit is correct, take to base units in Pa
            * Throws ValueError if unit is wrong
            * Assigning Pa units to scalar input
    """

    if isinstance(pressure, u.Quantity):
        if pressure.to_base_units().u == u.kg/u.m/u.s**2:
            pressure = pressure.to_base_units()
        else:
            raise ValueError(
                f"\n\t"
                f"Input {pressure} has unsupported units.\n\t"
                f"Input must have units of 'pascal' or equivalent;\n\t"
                f"otherwise, if dimensionless, it will\n\t"
                f"be assigned a Pa unit.\n"
            )
    else:
        pressure = u.Quantity(pressure, u.kg/u.m/u.s**2)

    return pressure

def in_molecular_weight(molec_wt):
    """ Handles molecular weight input

        Parameters:
            molec_wt    (float) [kg/mol | dimensionless]

        Returns:
            molec_wt    (float) [kg/mol]

        Notes:
            * If unit is correct, take to base units in kg/mol
            * Throws ValueError if unit is wrong
            * Assigning kg/mol units to scalar input
    """

    if isinstance(molec_wt, u.Quantity):
        if molec_wt.to_base_units().u == u.kg/u.mol:
            molec_wt = molec_wt.to_base_units()
        else:
            raise ValueError(
                f"\n\t"
                f"Input {molec_wt} has unsupported units.\n\t"
                f"Input must have molecular weight units of\n\t"
                f"'kg/mol';\n\t"
                f"otherwise, if dimensionless, it will.\n\t"
                f"be assigned a kg/mol unit.\n"
            )
    else:
        molec_wt = u.Quantity(molec_wt, u.kg / u.mol)

    return molec_wt
