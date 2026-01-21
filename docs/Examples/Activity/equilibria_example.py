"""Liquid-vapor partitioning equilibria example.

Demonstrates LiquidVaporPartitioningStrategy for computing
gas-particle equilibrium concentrations.

This example shows:
- Creating a liquid-vapor partitioning strategy
- Solving for equilibrium with organic species
- Interpreting the equilibrium results
"""

import numpy as np
import particula as par


def main():
    """Run equilibria partitioning example."""
    # Define organic species properties
    # Three species with different volatilities (C* values)
    c_star_j_dry = np.array([1e-6, 1e-4, 1e-2])  # saturation concentrations
    concentration_organic = np.array([1.0, 5.0, 10.0])  # ug/m^3
    molar_mass = np.array([200.0, 200.0, 200.0])  # g/mol
    o2c_ratio = np.array([0.2, 0.3, 0.5])  # O:C ratios
    density = np.array([1200.0, 1200.0, 1200.0])  # kg/m^3

    # 1. Create partitioning strategy at 75% RH
    strategy = par.equilibria.LiquidVaporPartitioningStrategy(
        water_activity=0.75,  # 75% relative humidity
    )

    # 2. Solve for equilibrium
    result = strategy.solve(
        c_star_j_dry=c_star_j_dry,
        concentration_organic_matter=concentration_organic,
        molar_mass=molar_mass,
        oxygen2carbon=o2c_ratio,
        density=density,
    )

    # 3. Print results
    print("=== Liquid-Vapor Partitioning ===")
    print(f"Water activity (RH): {strategy.water_activity * 100:.0f}%\n")

    print("Input organic species:")
    print(f"  C* (dry): {c_star_j_dry}")
    print(f"  Total concentration: {concentration_organic} ug/m^3")
    print(f"  O:C ratios: {o2c_ratio}")

    print(f"\n=== Equilibrium Results ===")
    print(f"Partition coefficients: {result.partition_coefficients}")
    print(f"  (fraction in condensed phase)")

    print(f"\nAlpha phase (water-rich):")
    print(
        f"  Species concentrations: {result.alpha_phase.species_concentrations}"
    )
    print(
        f"  Water concentration: {result.alpha_phase.water_concentration:.2f} ug/m^3"
    )
    print(
        f"  Total concentration: {result.alpha_phase.total_concentration:.2f} ug/m^3"
    )

    if result.beta_phase is not None:
        print(f"\nBeta phase (organic-rich):")
        print(
            f"  Species concentrations: "
            f"{result.beta_phase.species_concentrations}"
        )
        print(
            f"  Water concentration: "
            f"{result.beta_phase.water_concentration:.2f} ug/m^3"
        )

    print(
        f"\nWater content: alpha={result.water_content[0]:.2f}, "
        f"beta={result.water_content[1]:.2f} ug/m^3"
    )
    print(f"Optimization error: {result.error:.2e}")

    # 4. Show effect of RH on partitioning
    print("\n=== Effect of RH on Partitioning ===")
    rh_values = [0.3, 0.5, 0.75, 0.9]
    for rh in rh_values:
        strat = par.equilibria.LiquidVaporPartitioningStrategy(
            water_activity=rh,
        )
        res = strat.solve(
            c_star_j_dry=c_star_j_dry,
            concentration_organic_matter=concentration_organic,
            molar_mass=molar_mass,
            oxygen2carbon=o2c_ratio,
            density=density,
        )
        mean_partition = np.mean(res.partition_coefficients)
        print(
            f"RH={rh * 100:.0f}%: mean partition coeff = {mean_partition:.4f}, "
            f"water = {res.alpha_phase.water_concentration:.2f} ug/m^3"
        )


if __name__ == "__main__":
    main()
