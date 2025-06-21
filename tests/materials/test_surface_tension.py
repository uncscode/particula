import numpy as np
from particula.util.materials.surface_tension import get_surface_tension


def test_get_surface_tension_scalar():
    st = get_surface_tension("water", 298.15)
    assert 0.05 < st < 0.08             # ~0.072 N/m expected


def test_get_surface_tension_array():
    temps = np.array([280.0, 298.15, 320.0])
    st = get_surface_tension("water", temps)
    assert st.shape == temps.shape
    # Surface tension decreases with temperature
    assert np.all(st[1:] < st[:-1])
