"""Check the impactor conversion independently of the frozen reactor outputs."""

import numpy as np
from particula.particles.properties.slip_correction_module import (
    get_cunningham_slip_correction,
)
from scripts.straw_combustion.forward.observations import (
    _rescale_density,
    _slip_diameter_squared,
)


def test_slip_and_density_conversion_across_impactor_window():
    """At fixed aerodynamic size, density times d² Cc(d) is conserved."""
    diameter = np.array([30.0, 100.0, 500.0, 12700.0])

    def drag_term(d):
        return d**2 * get_cunningham_slip_correction(2 * 66.0 / d)

    np.testing.assert_allclose(
        _slip_diameter_squared(diameter), drag_term(diameter)
    )
    for density in (1980.0, 2000.0, 2660.0):
        mobility = _rescale_density(diameter, 1000.0 / density)
        np.testing.assert_allclose(
            density * drag_term(mobility),
            1000.0 * drag_term(diameter),
            rtol=1e-12,
        )
