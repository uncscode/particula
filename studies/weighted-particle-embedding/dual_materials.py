"""SI constants for the ideal two-vapor system."""

import numpy as np
from water_constants import SATURATION

NAMES = ("water", "semivolatile")
MOLAR = np.array([0.01801528, 0.12, 0.28, 0.06])
DENSITY = np.array([1000.0, 1200.0, 1400.0, 1400.0])
DIFFUSIVITY = np.array([2.4e-5, 7e-6])
ACCOMMODATION = np.array([0.20, 0.16])
TEMPERATURE = 293.0
R = 8.31446261815324
SURFACE_TENSION = 0.06
SURFACE_DENSITY = 1100.0
PURE_PRESSURE = np.array([SATURATION * R * TEMPERATURE / MOLAR[0], 12.0])
SATURATED = PURE_PRESSURE * MOLAR[:2] / (R * TEMPERATURE)
