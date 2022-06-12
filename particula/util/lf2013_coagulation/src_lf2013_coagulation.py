""" calculate ion--particle coagulation according to lf2013
"""

import os

import numpy as np


# flake8: noqa: C901
# pylint: disable=too-many-arguments, too-many-locals, too-many-branches
def lf2013_coag_full(
    ion_type="air",
    particle_type="conductive",
    temperature_val=298.15,
    pressure_val=101325,
    charge_vals=None,
    radius_vals=None,
):
    """ calculate ion--particle coagulation according to lf2013
    """

    if charge_vals is not None and (
            max(charge_vals) > 100 or min(charge_vals) < -100):
        raise ValueError("charge_vals must be between -100 and 100")

    if ion_type == "air" and particle_type == "conductive":
        if temperature_val == 298.15 and pressure_val == 101325:
            negfn = "S1.txt"
            posfn = "S2.txt"
        elif temperature_val == 218.15 and pressure_val == 4480:
            negfn = "S7.txt"
            posfn = "S8.txt"
        else:
            raise ValueError("Invalid combination")
    elif ion_type == "water" and particle_type == "conductive":
        if temperature_val == 298.15 and pressure_val == 101325:
            negfn = "S4.txt"
            posfn = "S5.txt"
        elif temperature_val == 218.15 and pressure_val == 4480:
            negfn = "S10.txt"
            posfn = "S11.txt"
        else:
            raise ValueError("Invalid combination")
    elif ion_type == "air" and particle_type == "polystyrene":
        if temperature_val != 298.15 and pressure_val != 101325:
            raise ValueError("Invalid combination")

        negfn = "S13.txt"
        posfn = "S14.txt"

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # expand dims to account for size
    negdata = np.expand_dims(
        np.loadtxt(os.path.join(dir_path, negfn), skiprows=1), axis=0)
    posdata = np.expand_dims(
        np.loadtxt(os.path.join(dir_path, posfn), skiprows=1), axis=0)

    coeffs = negdata.shape[-1]-5  # take 5 out (metdata)

    if isinstance(radius_vals, float):
        radius_vals = np.array([radius_vals])

    rads = np.expand_dims(
        np.expand_dims(
            radius_vals.squeeze(), axis=-1), axis=-1)
    powers = np.linspace(0, coeffs-1, coeffs)

    neg = 10**np.sum(
        (rads >= negdata[:, :, -2:-1]) *
        (rads <= negdata[:, :, -1:]) *
        negdata[:, :, 1:coeffs+1] *
        (np.log10(rads))**powers,
        axis=-1)

    pos = 10**np.sum(
        (rads >= posdata[:, :, -2:-1]) *
        (rads <= posdata[:, :, -1:]) *
        posdata[:, :, 1:coeffs+1] *
        (np.log10(rads))**powers,
        axis=-1)

    neg[neg == 1] = np.nan
    pos[pos == 1] = np.nan

    if isinstance(charge_vals, list):
        charges = [i+100 for i in charge_vals]
    else:
        charges = [charge_vals + 100]

    return neg[:, charges], pos[:, charges]
