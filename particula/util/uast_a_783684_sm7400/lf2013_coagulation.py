""" calculate ion--particle coagulation according to lf2013
"""

import numpy as np


def lf2013_coag_full(  # pylint: disable=too-many-arguments, too-many-locals
    ion_type="air",
    particle_type="conductive",
    temperature_val=298.15,
    pressure_val=101325,
    charge_vals=None,
    radius_vals=None,
):
    """ calculate ion--particle coagulation according to lf2013
    """
    if ion_type=="air" and particle_type=="conductive":
        if temperature_val==298.15 and pressure_val==101325:
            negfn = "S1.txt"
            posfn = "S2.txt"
        elif temperature_val==218.15 and pressure_val==4480:
            negfn = "S7.txt"
            posfn = "S8.txt"
        else:
            raise ValueError("Invalid combination")
    elif ion_type=="water" and particle_type=="conductive":
        if temperature_val==298.15 and pressure_val==101325:
            negfn = "S4.txt"
            posfn = "S5.txt"
        elif temperature_val==218.15 and pressure_val==4480:
            negfn = "S10.txt"
            posfn = "S11.txt"
        else:
            raise ValueError("Invalid combination")
    elif ion_type=="air" and particle_type=="polystyrene":
        if temperature_val != 298.15 and pressure_val != 101325:
            raise ValueError("Invalid combination")

        negfn = "S13.txt"
        posfn = "S14.txt"

    # expand dims to account for size
    negdata = np.expand_dims(
        np.loadtxt(negfn, skiprows=1), axis=0)
    posdata = np.expand_dims(
        np.loadtxt(posfn, skiprows=1), axis=0)

    coeffs = negdata.shape[-1]-5  # take 5 out (metdata)

    rads = np.expand_dims(
        np.expand_dims(
            radius_vals.squeeze(), axis=-1), axis=-1)
    powers = np.linspace(0, coeffs-1, coeffs)

    neg = 10**np.sum(
        (rads>=negdata[:, :, -2:-1]) *
        (rads<=negdata[:, :, -1:]) *
        negdata[:, :, 1:coeffs+1] *
        (np.log10(rads))**powers,
        axis=-1)

    pos = 10**np.sum(
        (rads>=posdata[:, :, -2:-1]) *
        (rads<=posdata[:, :, -1:]) *
        posdata[:, :, 1:coeffs+1] *
        (np.log10(rads))**powers,
        axis=-1)

    neg[neg==1] = np.nan
    pos[pos==1] = np.nan

    _ = charge_vals

    return neg, pos
