"""Kappa hygroscopic parameter activity example.

Demonstrates ActivityKappaParameter for computing water activity
in hygroscopic aerosol particles using kappa-Kohler theory.

The kappa parameter provides a single-parameter representation of
hygroscopicity, relating dry particle volume to water activity.

This example shows:
- Creating a kappa parameter activity strategy
- Computing water activity at different water contents
- Understanding kappa values for common aerosol species
"""

import numpy as np
import particula as par


def main():
    """Run kappa parameter activity example."""
    # Define species properties for water and ammonium sulfate
    # kappa=0 for water (not hygroscopic itself)
    # kappa=0.61 for ammonium sulfate (highly hygroscopic)
    kappa_values = np.array([0.0, 0.61])
    densities = np.array([1000.0, 1770.0])  # kg/m^3
    molar_masses = np.array([18.015e-3, 132.14e-3])  # kg/mol

    # 1. Create kappa parameter activity strategy
    strategy = par.particles.ActivityKappaParameter(
        kappa=kappa_values,
        density=densities,
        molar_mass=molar_masses,
        water_index=0,
    )

    # 2. Compute activity at different water contents
    print("=== Kappa Parameter Activity ===")
    print(f"Kappa values: {kappa_values}")
    print(f"Species: water (kappa=0), ammonium sulfate (kappa=0.61)\n")

    water_fractions = [0.3, 0.5, 0.7, 0.9]
    for water_frac in water_fractions:
        # Mass concentrations (water fraction, salt fraction)
        mass = np.array([water_frac, 1.0 - water_frac]) * 1e-9  # kg/m^3
        activity = strategy.activity(mass_concentration=mass)
        print(f"Water mass fraction: {water_frac:.1f}")
        print(f"  Water activity: {activity[0]:.4f}")
        print(f"  Salt activity:  {activity[1]:.4f}\n")

    # 3. Show how kappa affects water activity
    print("=== Effect of Kappa on Water Activity ===")
    print("(at 50% water by mass)\n")

    # Test different kappa values
    kappa_test_values = [0.0, 0.1, 0.3, 0.61, 1.0]
    for kappa in kappa_test_values:
        test_strategy = par.particles.ActivityKappaParameter(
            kappa=np.array([0.0, kappa]),
            density=np.array([1000.0, 1500.0]),
            molar_mass=np.array([18.015e-3, 100.0e-3]),
            water_index=0,
        )
        mass = np.array([0.5, 0.5]) * 1e-9
        activity = test_strategy.activity(mass_concentration=mass)
        print(f"kappa={kappa:.2f}: water activity = {activity[0]:.4f}")


if __name__ == "__main__":
    main()
