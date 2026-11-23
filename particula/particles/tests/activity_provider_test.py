"""Versioned native activity selection preserves custom pressure overrides."""

import numpy as np
from particula.particles.activity_strategies import ActivityIdealMass


class CustomPressure(ActivityIdealMass):
    """Provide a pressure override independent of the ideal activity."""

    def partial_pressure(self, pure_vapor_pressure, mass_concentration):
        """Use all three material lanes."""
        np.testing.assert_array_equal(pure_vapor_pressure, [1, 1, 1])
        return mass_concentration / 10


def test_custom_pressure_sees_all_material_before_selection():
    """Select the requested material order after the pressure call."""
    masses = np.array([[1, 3, 2], [4, 5, 6]], dtype=float)
    values = CustomPressure().neutral_activities(
        masses, 298.0, [2, 0], np.array([True, True])
    )
    np.testing.assert_array_equal(values, [[0.2, 0.1], [0.6, 0.4]])
