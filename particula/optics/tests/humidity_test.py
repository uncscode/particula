"""Water uptake for optical closure satisfies the original equilibrium."""

import numpy as np
import numpy.testing as npt
import pytest
from particula.optics.humidity import hydrate_index, subsaturated_growth
from scipy.optimize import brentq


def test_subsaturated_growth():
    """Compare with a direct wet-radius root across size and RH."""
    dry = np.geomspace(8e-9, 1e-6, 6)[None, :]
    rh = np.array([0.05, 0.6, 0.9, 0.999])[:, None]
    kappa = np.array([0.01, 0.15, 0.3, 0.5, 0.65, 1.2])[None, :]
    kelvin = 1.05e-9
    wet, water = subsaturated_growth(dry, rh, kappa, kelvin)
    r, humidity, k = np.broadcast_arrays(dry, rh, kappa)
    for index in np.ndindex(wet.shape):
        dry_i, rh_i, k_i = r[index], humidity[index], k[index]

        def equilibrium(growth, k_i=k_i, dry_i=dry_i, rh_i=rh_i):
            w = growth**3 - 1
            return w / (w + k_i) * np.exp(kelvin / (growth * dry_i)) - rh_i

        ratio = brentq(equilibrium, 1, np.cbrt(1 + k_i * rh_i / (1 - rh_i)))
        npt.assert_allclose(wet[index], dry_i * ratio, rtol=1e-12)
        npt.assert_allclose(equilibrium(wet[index] / dry_i), 0, atol=4e-14)
    _, flat = subsaturated_growth(dry, rh, kappa, 0.0)
    npt.assert_allclose(flat, kappa * rh / (1 - rh), rtol=5e-16)
    npt.assert_allclose(wet, dry * np.cbrt(1 + water), rtol=1e-15)


def test_dry_limits():
    """Dry and insoluble limits have zero uptake and unchanged index."""
    wet, water = subsaturated_growth(50e-9, [0, 0.9], 0.0, 1e-9)
    npt.assert_array_equal(wet, 50e-9)
    npt.assert_array_equal(water, 0)
    index = hydrate_index(1.6 + 0.025j, 1.33, water)
    npt.assert_allclose(index, 1.6 + 0.025j, rtol=1e-15)
    with pytest.raises(ValueError, match="growth coordinates"):
        subsaturated_growth(1e-7, 1, 0.3, 1e-9)
