"""Ideal activity calculation example.

Demonstrates using ActivityIdealMolar to compute activity
from mass concentrations using Raoult's Law.

This example shows:
- Creating an ideal molar activity strategy
- Computing activity from mass concentrations
- Computing partial pressures from pure vapor pressures
"""

import numpy as np
import particula as par


def main():
    """Run ideal activity example."""
    # 1. Create ideal activity strategy
    # Molar masses for water and an organic compound
    strategy = par.particles.ActivityIdealMolar(
        molar_mass=np.array([18.015e-3, 200.0e-3]),  # kg/mol: water, organic
    )

    # 2. Define mass concentrations
    # Equal mass of water and organic (50/50 by mass)
    mass = np.array([0.5e-9, 0.5e-9])  # kg/m^3

    # 3. Compute activity (mole fraction based for ideal mixing)
    activity = strategy.activity(mass_concentration=mass)

    # 4. Print results
    print("=== Ideal Activity (Raoult's Law) ===")
    print(f"Molar masses: {strategy.molar_mass * 1e3} g/mol")
    print(f"Mass concentrations: {mass * 1e9} ng/m^3")
    print(f"Activity values: {activity}")

    # 5. Compute partial pressures
    # Pure vapor pressures at 298 K (example values)
    pure_pressure = np.array([3169.0, 1e-3])  # Pa: water, organic
    partial_pressure = strategy.partial_pressure(
        pure_vapor_pressure=pure_pressure,
        mass_concentration=mass,
    )

    print(f"\nPure vapor pressures: {pure_pressure} Pa")
    print(f"Partial pressures: {partial_pressure} Pa")

    # 6. Show activity is mole fraction for ideal mixing
    # Calculate mole fractions directly for comparison
    moles = mass / strategy.molar_mass
    mole_fractions = moles / np.sum(moles)
    print(f"\nMole fractions: {mole_fractions}")
    print(
        f"Activity = mole fraction (ideal): {np.allclose(activity, mole_fractions)}"
    )


if __name__ == "__main__":
    main()
