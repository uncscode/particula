"""Non-ideal activity using BAT model.

Demonstrates ActivityNonIdealBinary for organic-water mixtures,
comparing ideal vs non-ideal activity coefficients.

The Binary Activity Thermodynamic (BAT) model accounts for non-ideal
interactions between water and organic species based on the organic's
oxygen-to-carbon ratio.

This example shows:
- Creating a non-ideal activity strategy via builder
- Comparing ideal vs non-ideal activity at various compositions
- Understanding the effect of O:C ratio on non-ideality
"""

import numpy as np
import particula as par


def main():
    """Run BAT activity comparison example."""
    # Define organic properties
    molar_mass_organic = 200.0e-3  # kg/mol
    o2c_ratio = 0.5  # oxygen to carbon ratio
    density_organic = 1200.0  # kg/m^3

    # 1. Create non-ideal (BAT) activity strategy via builder
    non_ideal = (
        par.particles.ActivityNonIdealBinaryBuilder()
        .set_molar_mass(molar_mass_organic, "kg/mol")
        .set_oxygen2carbon(o2c_ratio)
        .set_density(density_organic, "kg/m^3")
        .build()
    )

    # 2. Create ideal activity strategy for comparison
    ideal = par.particles.ActivityIdealMolar(
        molar_mass=np.array([18.015e-3, molar_mass_organic]),
    )

    # 3. Test at various organic mass fractions
    mass_fractions = np.array([0.2, 0.5, 0.8])
    print("=== Ideal vs Non-Ideal Activity Comparison ===")
    print(f"Organic O:C ratio: {o2c_ratio}")
    print(f"Organic molar mass: {molar_mass_organic * 1e3:.1f} g/mol\n")

    for org_frac in mass_fractions:
        # Mass concentrations: [water, organic]
        mass = np.array([1.0 - org_frac, org_frac]) * 1e-9  # kg/m^3

        ideal_activity = ideal.activity(mass_concentration=mass)
        non_ideal_activity = non_ideal.activity(mass_concentration=mass)

        print(f"Organic mass fraction: {org_frac:.1f}")
        print(
            f"  Mass conc: water={mass[0] * 1e9:.1f}, org={mass[1] * 1e9:.1f} ng/m^3"
        )
        print(
            f"  Ideal activity:     water={ideal_activity[0]:.4f}, "
            f"org={ideal_activity[1]:.4f}"
        )
        print(
            f"  Non-ideal (BAT):    water={non_ideal_activity[0]:.4f}, "
            f"org={non_ideal_activity[1]:.4f}"
        )

        # Activity coefficient = activity / mole fraction
        moles = mass / np.array([18.015e-3, molar_mass_organic])
        mole_frac = moles / np.sum(moles)
        gamma = non_ideal_activity / mole_frac
        print(
            f"  Activity coeffs:    water={gamma[0]:.4f}, org={gamma[1]:.4f}\n"
        )


if __name__ == "__main__":
    main()
