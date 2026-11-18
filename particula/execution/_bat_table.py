"""Tabulate homogeneous binary particle activities for device interpolation."""

import numpy as np


def _binary_activity_table(activity):
    """Tabulate both activities on a uniform organic mole-fraction grid."""
    molar = np.array([0.01801528, activity.molar_mass_kg])

    def evaluate(x):
        return activity.activity(np.column_stack((1.0 - x, x)) * molar)

    for power in range(8, 17):
        grid = np.linspace(0.0, 1.0, 2**power + 1)
        table = evaluate(grid)
        for fraction in (0.25, 0.5, 0.75):
            probe = grid[:-1] + fraction / (len(grid) - 1)
            interpolated = table[:-1] + fraction * np.diff(table, axis=0)
            if not np.all(np.abs(evaluate(probe) - interpolated) <= 1e-4):
                break
        else:
            return table.ravel()
    raise ValueError("BAT activity table exceeded its interpolation budget")
