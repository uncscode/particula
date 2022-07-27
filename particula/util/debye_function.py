""" debye function
"""
import numpy as np

from particula import u


def df1(var):
    """ debye function
    """
    if isinstance(var, u.Quantity):
        var = var.m
    xvar = np.linspace(0, var, 1000)
    return np.trapz(
        xvar[1:]/(np.exp(xvar[1:])-1),
        xvar[1:],
        axis=0
    )/var
