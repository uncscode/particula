# Activity Calculations

This section covers the theoretical foundations for activity calculations in particula, including the Binary Activity Thermodynamics (BAT) model and liquid-vapor partitioning. These concepts are essential for understanding how particula models organic aerosol thermodynamics and gas-particle equilibria.

Activity calculations determine how chemical species behave in mixtures, accounting for non-ideal interactions between molecules. In aerosol science, accurate activity models are critical for predicting:

- Water uptake by organic aerosols
- Phase separation in mixed organic-water systems
- Gas-particle partitioning of semi-volatile compounds
- Cloud droplet activation

## Topics

<div class="grid cards" markdown>

-   __[Activity Theory](activity_theory.md)__

    ---

    Raoult's Law, activity coefficients, and the BAT model for
    non-ideal organic-water mixtures. Learn how particula calculates
    thermodynamic activities using AIOMFAC-derived fits.

    [:octicons-arrow-right-24: Learn More](activity_theory.md)

-   __[Equilibria Theory](equilibria_theory.md)__

    ---

    Liquid-vapor partitioning, phase separation, and equilibrium
    solving in aerosol systems. Understand how particula solves
    for gas-particle equilibrium concentrations.

    [:octicons-arrow-right-24: Learn More](equilibria_theory.md)

</div>

## Related Resources

- **Examples**: See the [Equilibria How-To Guide](../../Examples/Equilibria/index.md) for practical tutorials
- **API Reference**: The `particula.activity` and `particula.equilibria` modules implement these concepts
