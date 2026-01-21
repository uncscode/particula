# Activity Examples

This section provides working code examples for computing chemical activity in aerosol particles using Particula's activity strategies. Learn how to use ideal, kappa-Kohler, and BAT (non-ideal) models through practical examples.

## Overview

Activity determines how species partition between gas and particle phases and controls water uptake in hygroscopic aerosols. These examples demonstrate:

- Computing activity from mass concentrations
- Comparing ideal vs. non-ideal activity models
- Using the kappa parameter for hygroscopic growth
- Integrating activity with equilibria calculations

## Example Scripts

<div class="grid cards" markdown>

-   __[Ideal Activity](ideal_activity_example.py)__

    ---

    Basic ideal activity calculation using Raoult's Law with
    `ActivityIdealMolar`. Good starting point for understanding
    the activity API.

-   __[BAT Model Activity](bat_activity_example.py)__

    ---

    Non-ideal activity using the Binary Activity Thermodynamic (BAT)
    model with `ActivityNonIdealBinary`. Compares ideal vs. non-ideal
    activity for organic-water mixtures.

-   __[Kappa Parameter Activity](kappa_activity_example.py)__

    ---

    Hygroscopic activity using kappa-Kohler theory with
    `ActivityKappaParameter`. Demonstrates water activity at
    different compositions for inorganic aerosols.

-   __[Equilibria Partitioning](equilibria_example.py)__

    ---

    Liquid-vapor partitioning using `LiquidVaporPartitioningStrategy`.
    Shows how activity integrates with equilibria calculations for
    gas-particle partitioning.

</div>

## Interactive Tutorial

For a comprehensive hands-on introduction, see the interactive Jupyter notebook:

[:octicons-arrow-right-24: Activity Tutorial](activity_tutorial.ipynb)

This tutorial covers:

- What is activity and why it matters for aerosol modeling
- Ideal activity with Raoult's Law
- Non-ideal activity with the BAT model
- Kappa parameter for hygroscopic growth
- Integration with equilibria calculations
- Visualization of activity vs. composition

## Related Resources

- **Existing Tutorial**: [Activity Tutorial (Particle Phase)](../Particle_Phase/Notebooks/Activity_Tutorial.ipynb) - Covers Strategy, Builder, and Factory patterns
- **Feature Guide**: [Activity System](../../Features/activity_system.md) - Comprehensive feature documentation
- **Theory**: [Activity Theory](../../Theory/Activity_Calculations/activity_theory.md) - Mathematical foundations

## Quick Reference

### Available Strategies

| Strategy | Import | Description |
|----------|--------|-------------|
| `ActivityIdealMolar` | `par.particles.ActivityIdealMolar` | Raoult's Law (mole fractions) |
| `ActivityIdealMass` | `par.particles.ActivityIdealMass` | Ideal (mass fractions) |
| `ActivityKappaParameter` | `par.particles.ActivityKappaParameter` | Kappa-Kohler model |
| `ActivityNonIdealBinary` | `par.particles.ActivityNonIdealBinary` | BAT non-ideal model |

### Basic Usage Pattern

```python
import numpy as np
import particula as par

# 1. Create strategy
strategy = par.particles.ActivityIdealMolar(
    molar_mass=np.array([18.015e-3, 200.0e-3]),  # kg/mol
)

# 2. Define mass concentrations
mass = np.array([0.5e-9, 0.5e-9])  # kg/m^3

# 3. Compute activity
activity = strategy.activity(mass_concentration=mass)
```
