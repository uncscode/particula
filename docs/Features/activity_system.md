# Activity System

> Strategy-based activity calculations for aerosol thermodynamics using ideal, kappa-Kohler, and BAT models.

## Overview

The activity system provides a unified interface for computing chemical activity in aerosol particles. Activity determines how species partition between gas and particle phases, controls water uptake and hygroscopic growth, and drives phase equilibria in organic-water mixtures.

This feature is built around user-facing APIs exposed via `particula.particles`:

- `ActivityIdealMolar`, `ActivityIdealMass` - Ideal activity strategies based on Raoult's Law
- `ActivityKappaParameter` - Kappa hygroscopic parameter model for water activity
- `ActivityNonIdealBinary` - BAT model for non-ideal organic-water mixtures
- `ActivityIdealMolarBuilder`, `ActivityIdealMassBuilder`, etc. - Validated, unit-aware builders
- `ActivityFactory` - Factory for selecting strategies by name

Integration with equilibria calculations is provided through `particula.equilibria`:

- `LiquidVaporPartitioningStrategy` - Liquid-vapor equilibrium using activity
- `Equilibria` - Runnable wrapper for equilibria strategies

## Key Benefits

- **Consistent dynamics workflow**: Use the same strategy-based API as condensation, coagulation, and wall loss
- **Multiple activity models**: Choose from ideal, kappa-Kohler, or BAT (non-ideal) models based on your system
- **Builder/factory validation**: Configure activity using unit-aware builders with automatic conversion and validation
- **Equilibria integration**: Combine activity strategies with liquid-vapor partitioning for thermodynamic equilibrium
- **Extensible design**: Add new activity models by subclassing `ActivityStrategy`

## Who It's For

This feature is designed for:

- **Aerosol modelers**: Computing water activity for hygroscopic growth and cloud droplet activation
- **Chamber experiment analysts**: Modeling organic aerosol partitioning with realistic thermodynamics
- **Model developers**: Implementing custom activity models while reusing particula's infrastructure
- **Researchers**: Comparing ideal vs. non-ideal activity effects on aerosol behavior

## Capabilities

### Unified activity API in `particula.particles`

Activity strategies are exposed alongside other particle components:

```python
import particula as par

# Strategy classes
par.particles.ActivityIdealMolar
par.particles.ActivityIdealMass
par.particles.ActivityKappaParameter
par.particles.ActivityNonIdealBinary

# Builder classes
par.particles.ActivityIdealMolarBuilder
par.particles.ActivityIdealMassBuilder
par.particles.ActivityKappaParameterBuilder
par.particles.ActivityNonIdealBinaryBuilder

# Factory class
par.particles.ActivityFactory
```

All activity strategies share a common interface:

- Call `activity(mass_concentration)` to compute activity from mass concentrations
- Call `partial_pressure(pure_vapor_pressure, mass_concentration)` to compute partial pressures

### Available strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `ActivityIdealMolar` | Raoult's Law based on mole fractions | Simple mixtures, dilute solutions |
| `ActivityIdealMass` | Ideal activity based on mass fractions | Mass-based calculations |
| `ActivityKappaParameter` | Kappa hygroscopic parameter model | Hygroscopic growth, CCN activation |
| `ActivityNonIdealBinary` | BAT model for non-ideal mixtures | Organic-water aerosol thermodynamics |

### Ideal activity strategies

For systems where activity coefficients are approximately 1, use the ideal strategies:

```python
import numpy as np
import particula as par

# Ideal molar activity (Raoult's Law)
strategy = par.particles.ActivityIdealMolar(
    molar_mass=np.array([18.015e-3, 200.0e-3]),  # kg/mol: water, organic
)

# Mass concentrations (kg/m^3)
mass = np.array([0.5e-9, 0.5e-9])

# Compute activity (returns mole fractions for ideal mixing)
activity = strategy.activity(mass_concentration=mass)
```

### Kappa parameter activity

For hygroscopic aerosols, the kappa parameter provides a single-parameter representation of water activity:

```python
import numpy as np
import particula as par

# Kappa parameter strategy
strategy = par.particles.ActivityKappaParameter(
    kappa=np.array([0.0, 0.61]),  # water, ammonium sulfate
    density=np.array([1000.0, 1770.0]),  # kg/m^3
    molar_mass=np.array([18.015e-3, 132.14e-3]),  # kg/mol
    water_index=0,
)

# Compute water activity at given composition
mass = np.array([0.7e-9, 0.3e-9])  # 70% water, 30% salt
activity = strategy.activity(mass_concentration=mass)
# activity[0] gives water activity
```

### Non-ideal activity (BAT model)

For organic-water mixtures with significant non-ideality, use the BAT model:

```python
import particula as par

# Build non-ideal activity strategy
strategy = (
    par.particles.ActivityNonIdealBinaryBuilder()
    .set_molar_mass(200.0e-3, "kg/mol")
    .set_oxygen2carbon(0.5)
    .set_density(1200.0, "kg/m^3")
    .build()
)

# Compute activity
mass = np.array([0.5e-9, 0.5e-9])  # water, organic
activity = strategy.activity(mass_concentration=mass)
```

### Builder usage patterns

Builders provide validated configuration with automatic unit conversion:

```python
import numpy as np
import particula as par

# Molar activity with unit conversion
molar_activity = (
    par.particles.ActivityIdealMolarBuilder()
    .set_molar_mass(np.array([18.015, 200.0]), "g/mol")  # converts to kg/mol
    .build()
)

# Kappa activity with all parameters
kappa_activity = (
    par.particles.ActivityKappaParameterBuilder()
    .set_kappa(np.array([0.0, 0.61]))
    .set_density(np.array([1.0, 1.77]), "g/cm^3")  # converts to kg/m^3
    .set_molar_mass(np.array([18.015, 132.14]), "g/mol")
    .set_water_index(0)
    .build()
)
```

### Factory usage for dynamic selection

The factory enables runtime strategy selection by name:

```python
import numpy as np
import particula as par

factory = par.particles.ActivityFactory()

# Select ideal molar strategy
ideal = factory.get_strategy(
    strategy_type="ideal_molar",
    parameters={
        "molar_mass": np.array([18.015e-3, 200.0e-3]),
    },
)

# Select kappa strategy
kappa = factory.get_strategy(
    strategy_type="kappa",
    parameters={
        "kappa": np.array([0.0, 0.3]),
        "density": np.array([1000.0, 1500.0]),
        "density_units": "kg/m^3",
        "molar_mass": np.array([18.015, 342.29]),
        "molar_mass_units": "g/mol",
        "water_index": 0,
    },
)
```

### Integration with equilibria

Activity strategies integrate with liquid-vapor partitioning calculations:

```python
import numpy as np
import particula as par

# Create partitioning strategy
partitioning = par.equilibria.LiquidVaporPartitioningStrategy(
    water_activity=0.75,  # 75% RH
)

# Solve for equilibrium
result = partitioning.solve(
    c_star_j_dry=np.array([1e-6, 1e-4, 1e-2]),  # saturation concentrations
    concentration_organic_matter=np.array([1.0, 5.0, 10.0]),  # ug/m^3
    molar_mass=np.array([200.0, 200.0, 200.0]),  # g/mol
    oxygen2carbon=np.array([0.2, 0.3, 0.5]),
    density=np.array([1200.0, 1200.0, 1200.0]),  # kg/m^3
)

# Access results
print(f"Partition coefficients: {result.partition_coefficients}")
print(f"Alpha phase water: {result.alpha_phase.water_concentration}")
```

## Getting Started

### Quick start: ideal activity calculation

```python
import numpy as np
import particula as par

# 1. Create activity strategy
strategy = par.particles.ActivityIdealMolar(
    molar_mass=np.array([18.015e-3, 200.0e-3]),  # kg/mol
)

# 2. Define mass concentrations
mass = np.array([0.5e-9, 0.5e-9])  # kg/m^3

# 3. Compute activity
activity = strategy.activity(mass_concentration=mass)
print(f"Activity: {activity}")

# 4. Compute partial pressures
pure_pressure = np.array([3169.0, 1e-3])  # Pa
partial_p = strategy.partial_pressure(
    pure_vapor_pressure=pure_pressure,
    mass_concentration=mass,
)
print(f"Partial pressures: {partial_p}")
```

### Prerequisites

- `particula` version 0.2.6 or later installed
- Basic familiarity with NumPy arrays
- Understanding of aerosol thermodynamics (helpful but not required)

## Typical Workflows

### 1. Choose an activity model

Select the appropriate strategy based on your system:

- **Simple mixtures**: Use `ActivityIdealMolar` or `ActivityIdealMass`
- **Hygroscopic aerosols**: Use `ActivityKappaParameter` with known kappa values
- **Organic-water mixtures**: Use `ActivityNonIdealBinary` (BAT model)

### 2. Configure with a builder

Use builders for validated configuration:

```python
strategy = (
    par.particles.ActivityNonIdealBinaryBuilder()
    .set_molar_mass(200.0, "g/mol")
    .set_oxygen2carbon(0.4)
    .set_density(1.2, "g/cm^3")
    .build()
)
```

### 3. Compute activity or partial pressure

Call the strategy methods with mass concentration arrays:

```python
activity = strategy.activity(mass_concentration=mass)
partial_p = strategy.partial_pressure(pure_vapor_pressure, mass_concentration=mass)
```

### 4. Integrate with equilibria

Combine with partitioning strategies for full thermodynamic calculations:

```python
partitioning = par.equilibria.LiquidVaporPartitioningStrategy(
    water_activity=target_rh,
)
result = partitioning.solve(...)
```

## Use Cases

### Use case 1: Hygroscopic growth calculation

**Scenario:** Calculate water activity for ammonium sulfate aerosol at different water contents.

**Solution:** Use `ActivityKappaParameter` with known kappa values and compute activity across a range of compositions.

### Use case 2: Organic aerosol partitioning

**Scenario:** Model gas-particle partitioning of organic compounds using realistic non-ideal thermodynamics.

**Solution:** Use `ActivityNonIdealBinary` with BAT model parameters, then integrate with `LiquidVaporPartitioningStrategy` for equilibrium calculations.

### Use case 3: Comparing activity models

**Scenario:** Evaluate the effect of non-ideal activity on partitioning predictions.

**Solution:** Create both ideal and non-ideal strategies, compute activity at the same compositions, and compare results.

## Configuration

### Strategy Parameters

| Strategy | Parameter | Description | Units |
|----------|-----------|-------------|-------|
| `ActivityIdealMolar` | `molar_mass` | Species molar masses | kg/mol |
| `ActivityIdealMass` | (none) | No parameters required | - |
| `ActivityKappaParameter` | `kappa` | Hygroscopic parameters | dimensionless |
| | `density` | Species densities | kg/m^3 |
| | `molar_mass` | Species molar masses | kg/mol |
| | `water_index` | Index of water species | int |
| `ActivityNonIdealBinary` | `molar_mass` | Organic molar mass | kg/mol |
| | `oxygen2carbon` | O:C ratio | dimensionless |
| | `density` | Organic density | kg/m^3 |

### Factory Strategy Types

| Type Name | Strategy Class |
|-----------|---------------|
| `"ideal_mass"` | `ActivityIdealMass` |
| `"ideal_molar"` | `ActivityIdealMolar` |
| `"kappa"` | `ActivityKappaParameter` |
| `"non_ideal_binary"` | `ActivityNonIdealBinary` |

## Best Practices

1. **Match model to system**: Use ideal strategies for simple mixtures, kappa for inorganic hygroscopic aerosols, and BAT for organic-water systems
2. **Use builder validation**: Set parameters through builders to ensure unit conversion and validity checks
3. **Check activity ranges**: Activity values should be in [0, 1] for condensed phases
4. **Consider temperature effects**: The BAT model includes temperature-dependent parameters; ensure consistency
5. **Validate with experiments**: Compare model predictions against measured hygroscopic growth or partitioning data
6. **Use consistent units**: Mass concentrations are expected in kg/m^3; use builders for automatic conversion

## Limitations

- **Binary organic-water only**: `ActivityNonIdealBinary` assumes a binary organic-water system; multi-organic mixtures use mean properties
- **No solid-liquid equilibria**: Current strategies handle liquid-vapor systems only
- **Temperature range**: BAT model parameters are optimized for ambient temperatures (250-310 K)
- **Kappa model assumptions**: Assumes volume additivity and spherical particles

## Related Documentation

- **Theory**: [Activity Theory](../Theory/Activity_Calculations/activity_theory.md) - Mathematical foundations
- **Theory**: [Equilibria Theory](../Theory/Activity_Calculations/equilibria_theory.md) - Partitioning equations
- **Examples**: [Activity Examples](../Examples/Activity/index.md) - Working code examples
- **Tutorial**: [Activity Tutorial](../Examples/Particle_Phase/Notebooks/Activity_Tutorial.ipynb) - Interactive notebook

## FAQ

### Which activity model should I use?

- **For dilute aqueous solutions**: `ActivityIdealMolar` (Raoult's Law)
- **For hygroscopic inorganics**: `ActivityKappaParameter` with measured kappa values
- **For organic aerosols**: `ActivityNonIdealBinary` (BAT model)

### How do I get activity coefficients?

Activity = activity coefficient x mole fraction. For ideal strategies, the activity coefficient is 1. For non-ideal strategies, you can compute activity coefficients by dividing activity by mole fraction.

### Can I use custom activity models?

Yes. Subclass `ActivityStrategy` from `particula.particles.activity_strategies` and implement the `activity()` and `partial_pressure()` methods.

### How does activity connect to equilibria?

Activity determines the effective concentration that drives phase equilibrium. The `LiquidVaporPartitioningStrategy` uses BAT model activity internally to compute gas-particle partitioning.

## See Also

- [Equilibria Examples](../Examples/Equilibria/index.md)
- [Particle Phase Tutorial](../Examples/Particle_Phase/index.md)
- [Wall Loss Strategy System](wall_loss_strategy_system.md)
