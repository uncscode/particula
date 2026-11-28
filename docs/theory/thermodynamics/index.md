# Aerosol thermodynamics

Activity and phase equilibrium determine how water and organic vapors partition into particles. In a rising air parcel, cooling changes saturation while condensation releases heat. Follow the derivation for the quantity you want to predict.

Activity calculations determine how chemical species behave in mixtures, accounting for non-ideal interactions between molecules. In aerosol science, accurate activity models are critical for predicting:

- Water uptake by organic aerosols
- Phase separation in mixed organic-water systems
- Gas-particle partitioning of semi-volatile compounds
- Cloud droplet activation

## Topics

<div class="grid cards" markdown>

-   __[Water, aerosol, and motion in a parcel](parcel-dynamics.md)__

    ---

    Derive the water budget of a rising parcel and follow gas and particles
    across a shared face.

-   __[κ-Köhler activation](kappa-kohler-activation.md)__

    ---

    Find the critical dry diameter for a prescribed water supersaturation and
    derive the equilibrium CCN number concentration of a mixed population.

-   __[Activity Theory](activity-theory.md)__

    ---

    Raoult's Law, activity coefficients, and the BAT model for
    non-ideal organic-water mixtures. Learn how particula calculates
    thermodynamic activities using AIOMFAC-derived fits.

    [:octicons-arrow-right-24: Learn More](activity-theory.md)

-   __[Equilibria Theory](equilibria-theory.md)__

    ---

    Liquid-vapor partitioning, phase separation, and equilibrium
    solving in aerosol systems. Understand how particula solves
    for gas-particle equilibrium concentrations.

    [:octicons-arrow-right-24: Learn More](equilibria-theory.md)

</div>

## Related Resources

- **Examples**: See the [Equilibria How-To Guide](../../studies/examples/thermodynamics/equilibria/index.md) for practical tutorials
- **API Reference**: The `particula.activity` and `particula.equilibria` modules implement these concepts
