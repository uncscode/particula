""" calculate the wall loss coefficient
"""
import numpy as np

from particula import u
from particula.util.input_handling import in_handling
from particula.util.debye_function import df1
from particula.util.settling_velocity import psv
from particula.util.diffusion_coefficient import pdc


def wlc(
    approx="none",
    ktp_val=0.1 * u.s**-1,
    pdc_val=None,
    crad=None,
    psv_val=None,
    **kwargs
):
    """ calculate the dilution loss coefficient
    """

    if approx == "none":
        return 0.0

    if approx == "simple":
        ktp_val = in_handling(ktp_val, u.s**-1)
        pdc_val = in_handling(
            pdc_val, u.m**2 / u.s) if pdc_val is not None else (
                pdc(**kwargs)
            )
        psv_val = in_handling(
            psv_val, u.m / u.s) if psv_val is not None else (
                psv(**kwargs)
        )
        crad = in_handling(
            crad, u.m) if crad is not None else (
                1 * u.m
        )

        return (
            6 * np.sqrt(ktp_val * pdc_val) / (np.pi * crad) *
            df1(
                np.pi * psv_val / (2 * np.sqrt(ktp_val * pdc_val))
            ) +
            psv_val / (4 * crad / 3)
        )

    return 0
