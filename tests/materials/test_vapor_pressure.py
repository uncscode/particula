import numpy as np
from particula.util.materials.vapor_pressure import get_vapor_pressure


def test_get_vapor_pressure_scalar():
    vp = get_vapor_pressure("water", 298.15)
    assert 2_000.0 < vp < 40_000.0      # ~3.2 kPa expected


def test_get_vapor_pressure_array():
    temps = np.array([280.0, 298.15, 320.0])
    vp = get_vapor_pressure("water", temps)
    assert vp.shape == temps.shape
    # Vapor pressure increases with temperature
    assert np.all(vp[1:] > vp[:-1])
