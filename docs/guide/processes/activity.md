# Activity System

> Strategy-based activity calculations for aerosol thermodynamics using ideal, kappa-Kohler, BAT, and optional AIOMFAC models.

## Overview

The activity system provides a unified interface for computing chemical activity in aerosol particles. Activity determines how species partition between gas and particle phases, controls water uptake and hygroscopic growth, and drives phase equilibria in organic-water mixtures.

This feature is built around user-facing APIs exposed via `particula.particles`:

- `ActivityIdealMolar`, `ActivityIdealMass` - Ideal activity strategies based on Raoult's Law
- `ActivityKappaParameter` - Kappa hygroscopic parameter model for water activity
- `ActivityNonIdealBinary` - organic activity in a binary water–organic mixture
- `ActivityBAT` - both particle-phase activities for kinetic water and organic exchange
- `ActivityIdealMolarBuilder`, `ActivityIdealMassBuilder`, etc. - Validated, unit-aware builders
- `ActivityFactory` - Factory for selecting strategies by name

Integration with equilibria calculations is provided through `particula.equilibria`:

- `LiquidVaporPartitioningStrategy` - Liquid-vapor equilibrium using activity
- `Equilibria` - Runnable wrapper for equilibria strategies

### Binary BAT in a kinetic simulation

Pass `ActivityBAT(molar_mass=0.2, oxygen2carbon=0.5, density=1243.493333)` as the condensation strategy's `activity_strategy` for the miscible reference organic. The molar mass is in kg/mol and the density in kg/m³. Particle materials must be ordered as water then organic. The gas species map can put either vapor in a different lane or leave a material involatile. At each rate evaluation the current particle masses determine the two activities. Kelvin curvature and the pure-species vapor pressures then determine the surface pressures that drive kinetic transfer.

Use this profile for a single homogeneous liquid. The model supplies a constrained liquid curve even at descriptors that permit demixing. A phase-separated droplet needs phase amounts and a choice of surface composition. The CPU evaluates the binary function directly and the GPU uses a prepared one-dimensional table. [Matched CPU and CUDA humidity change](../../studies/benchmarks/bat-humidity-pulse/bat_humidity_pulse.py) · [Reference curves and supported phase assumptions](../../studies/validation/material-properties/binary-bat.md)

### AIOMFAC in native isothermal condensation

The development extra `particula[aiomfac]` pairs Particula with the separate
`particula-aiomfac` distribution. Select `ActivityFactory().get_strategy("aiomfac", parameters)`
with a backend `MixtureSpec`, a `material_mapping` from native material indices
to component names, and any `inert_indices`. Declare all dissolved organics
and fully dissociated salts, including those that stay in the particle. Use the
gas species map to select the exchanging neutral material. Each CPU rate call
uses its current liquid composition and temperature in kelvin.

The standalone calculator returns neutral activities on the mole-fraction
scale and ion coefficients on the molality scale. Systems that activate
bisulfate or closed carbonate equilibrium require a speciation solver.
The backend source distribution includes the executable closed-water example
and instructions for installing the paired development wheels.

### Stable liquids and gas–particle equilibrium

Select `phase_mode="stable"` in the AIOMFAC activity factory parameters to
resolve internal liquid separation during condensation or a parcel calculation.
The default for simulations is `"single_liquid"`. Stable mode uses direct
activity evaluations at the current particle composition and temperature.
Internal equilibration is instantaneous. The particle retains one total mass
per material. At every gas exchange step, coexisting liquids give the same
neutral chemical potential. The transport model applies its own curvature term.

Configure the provider for water, 2-propanol and ammonium sulfate in native
material order. Use the same provider for a fixed condensed inventory and
for gas–liquid partitioning.
The first two calls accept masses in any common unit. The bulk solver accepts
analytical concentrations and pure-vapor saturation concentrations in µg/m³.

```python title="Requires particula-aiomfac"
from particula.equilibria import EquilibriaFactory
from particula.particles import ActivityFactory
from particula_aiomfac import MixtureSpec

groups = {
    "water": {16: 1},
    "2-propanol": {141: 2, 151: 1, 153: 1},
    "ammonium sulfate": {204: 2, 261: 1},
}
activity = ActivityFactory().get_strategy("aiomfac", {
    "mixture": MixtureSpec(groups),
    "material_mapping": dict(enumerate(groups)),
    "phase_mode": "stable",
})
liquids = activity.phase_equilibrium([0.40, 0.42, 0.18], 303.15)
water_activity = activity.activities_at_temperature(
    [0.40, 0.42, 0.18], 303.15, ["water"]
)
strategy = EquilibriaFactory().get_strategy(
    "liquid_vapor", {"activity_model": activity}
)
result = strategy.solve(
    inventories=[0.0, 7.0, 3.0], temperature=303.15,
    saturation_concentrations={"water": 3.03e7}, relative_humidity=0.77,
)
```

A specified relative humidity is a ratio. The reported water exchange is
positive for net uptake from that reservoir into the combined gas and liquid
system. Omit `relative_humidity` to close the supplied total water inventory.
List each exchanging neutral in `saturation_concentrations`, including water.
Unlisted components remain condensed. The result provides phase amounts,
per-phase chemistry, gas inventories, water exchange, and convergence diagnostics.
When `result.liquids` is `None`, inspect `result.solid_liquid` for retained
crystals or dry ions. Without condensed material all mass lies in the gas.

Standalone equilibrium defaults to stable liquids. Its `execution="cuda"`
option uses device liquid calculations within a host-controlled bulk reservoir
solve. Native resident condensation and parcel evaluation keep the particle
phase search on the device. The search admits at most two liquids and requires
a dissolved mixture. Inspect reported solver failures at unsupported states.
[Derivation and equilibrium assumptions](../../theory/thermodynamics/equilibria-theory.md)
· [Humidity sweep](../../studies/examples/thermodynamics/equilibria/aiomfac_humidity_sweep.py)
· [Closed evaporating droplet](../../studies/examples/thermodynamics/equilibria/aiomfac_evaporating_droplet.py)
· [Humidity ramp for 1,000 individual droplets](../../studies/examples/thermodynamics/equilibria/aiomfac_mixing_state_ramp.py)

### Select the carbonate reference

For stable carbonate calculations, pass `"carbonate_model": "gibbs_consistent"`
in the activity factory parameters. This extension uses a molal reference for
dissolved CO₂ and includes the corresponding ion and solvent chemical
potentials. For an explicitly mapped CO₂ lane, use its Henry-law pressure
at unit molal activity when converting activities to partial pressures.
Supply gas equilibrium constants on that same reference. Historical
single-liquid calculations use the default `"pinned"` carbonate model.
Both phase stability and the carbonate extension require direct evaluation.
[Carbonate derivation](../../theory/thermodynamics/equilibria-theory.md)

### Follow salt mass into dissolved ions and back into crystals

`SpeciesInventory` holds **physical mass concentrations** in µg/m³ of air.
The solid, liquid-neutral, liquid-ion, gas, surface-ion, and dry-residual-ion
fields are disjoint. To supply solid NaCl, put its mass in `solids`. Dissolved
Na⁺ and Cl⁻ use AIOMFAC IDs 202 and 242 in `liquid_ions`. To supply an
already dissolved sample, put its mass directly into those ion fields.

```python title="Requires particula-aiomfac"
from particula.particles import ActivityFactory
from particula_aiomfac import MixtureSpec, SpeciesInventory

groups = {"water": {16: 1}, "NaCl": {202: 1, 242: 1}}
salt_model = ActivityFactory().get_strategy("aiomfac", {
    "mixture": MixtureSpec(groups),
    "material_mapping": dict(enumerate(groups)),
    "phase_mode": "stable",
    "solids": {"NaCl": "halite"},
})
state = salt_model.equilibrate_species(
    species_inventory=SpeciesInventory(
        liquid_neutrals={"water": 1000.0}, solids={"NaCl": 1.0},
    ),
    temperature=298.15,
)
print(state.inventory.solids, state.inventory.liquid_ions)
```

At 298.15 K the 1 µg/m³ of halite fully dissolves in this example, yielding
0.393375 µg/m³ of sodium and 0.606625 µg/m³ of chloride. With 1 µg/m³
of water at the same salt loading, about 0.639611 µg/m³ stays crystalline.
At complete dry-out an ion without sufficient counterion or a specified mineral
remains in `residual_ions`. Use `inventory.total_mass` for the conserved mass.
The analytical feed and the per-phase species diagnostics are alternative
descriptions of that material.

For a finite gas inventory, add `gas_reactions` to the call. The supplied
`HCL_298` and `NH3_298` reactions in `particula_aiomfac.reservoirs` apply at
298.15 K. Declare their aqueous constituents in the mixture. A
`GasReaction` provides the stoichiometry, natural log of the equilibrium
constant, temperature, reference pressure, and signed gas charge. Charged
transfer also requires an `IonSurface` with measured or explicitly assumed
binding constants, site capacity in mol/m³ air, and capacitance in F/m³ air.
The returned interfacial and gas populations close the charge balance.
The equilibrium amount of a gas ion depends on that electrochemical input.

To apply a closed solve to native arrays, pass
`species_layout=SpeciesLayout(...)` and `species_options={...}` to
`Equilibria(strategy, ...)`. Map each physical material exactly once, including
solid and dissolved ions as separate materials. Select one internally mixed
population per box. Two reserved slots allow the runnable to represent a
fractional mean charge with adjacent integer charge states. The resulting
per-box reports are on `aerosol.equilibria_result`. Native kinetic condensation
and parcel transport exchange the configured neutral vapors. The species
runnable applies ion, crystal and charged-gas transformations as an
instantaneous equilibrium step.

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
par.particles.ActivityBAT

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
    strategy_type="molar_ideal",
    parameters={
        "molar_mass": np.array([18.015e-3, 200.0e-3]),
        "molar_mass_units": "kg/mol",
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

Activity strategies integrate with liquid-vapor partitioning calculations.
This equilibrium interface uses matching µg/m³ concentration inputs and g/mol
molar masses; activity strategies above use kg/mol. Water supplied by the
prescribed humidity reservoir is separate from the organic species balance:

```python
import numpy as np
import particula as par

# Create partitioning strategy
partitioning = par.equilibria.LiquidVaporPartitioningStrategy(
    water_activity=0.75,  # 75% RH
)

# Solve for equilibrium
result = partitioning.solve(
    c_star_j_dry=np.array([1e-6, 1e-4, 1e-2]),  # ug/m³, matching organic inventory units
    concentration_organic_matter=np.array([1.0, 5.0, 10.0]),  # ug/m^3
    molar_mass=np.array([200.0, 200.0, 200.0]),  # g/mol
    oxygen2carbon=np.array([0.2, 0.3, 0.5]),
    density=np.array([1200.0, 1200.0, 1200.0]),  # kg/m^3
)

# Access results
print(f"Partition coefficients: {result.partition_coefficients}")
print(f"Alpha phase water: {result.alpha_phase.water_concentration}")
```

Run the Python snippets in page order so imports and example arrays are
available to the later workflow snippets. These functional calculations do
not construct or mutate an `Aerosol`.

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
partial_p = strategy.partial_pressure(pure_pressure, mass_concentration=mass)
```

### 4. Integrate with equilibria

Combine with partitioning strategies for full thermodynamic calculations:

```python
partitioning = par.equilibria.LiquidVaporPartitioningStrategy(
    water_activity=0.75,
)
result = partitioning.solve(
    c_star_j_dry=np.array([1e-6, 1e-4, 1e-2]),  # ug/m³
    concentration_organic_matter=np.array([1.0, 5.0, 10.0]),  # ug/m³
    molar_mass=np.array([200.0, 200.0, 200.0]),  # g/mol
    oxygen2carbon=np.array([0.2, 0.3, 0.5]),
    density=np.array([1200.0, 1200.0, 1200.0]),  # kg/m³
)
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
3. **Check finite, nonnegative activities**: Non-ideal activity can exceed one; do not clip it to a mole-fraction bound.
4. **Consider temperature effects**: The BAT coefficients were fitted at 298.15 K. Specify the temperature used for vapor pressure separately
5. **Validate with experiments**: Compare model predictions against measured hygroscopic growth or partitioning data
6. **Use consistent units**: Mass concentrations are expected in kg/m^3; use builders for automatic conversion

## Limitations

- **Binary organic-water only**: `ActivityNonIdealBinary` assumes a binary organic-water system. Its output is organic activity. Use `ActivityBAT` when condensation requires both species activities
- **No solid-liquid equilibria**: Current strategies handle liquid-vapor systems only
- **Temperature range**: The published fit uses 298.15 K training data. See [activity theory](../../theory/thermodynamics/activity-theory.md) for temperature assumptions
- **Kappa model assumptions**: Assumes volume additivity and spherical particles

## Related Documentation

- **Theory**: [Activity Theory](../../theory/thermodynamics/activity-theory.md) - Mathematical foundations
- **Theory**: [Equilibria Theory](../../theory/thermodynamics/equilibria-theory.md) - Partitioning equations
- [Water activity calculations](../../studies/examples/thermodynamics/activity/index.md) compare ideal mixing, κ, and BAT assumptions.
- [Construct an activity model](../../studies/examples/thermodynamics/particle-properties/activity_tutorial.py) using strategies, builders, and factories in Python.

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

Activity determines the effective concentration that drives phase equilibrium. `LiquidVaporPartitioningStrategy` uses BAT by default. Supply a configured AIOMFAC provider as `activity_model` for mixture-specific equilibrium.

## See Also

- [Designing a size-resolved CCN experiment](ccn-retrieval.md)
- [Equilibria Examples](../../studies/examples/thermodynamics/equilibria/index.md)
- [Particle Phase Tutorial](../../studies/tutorials/building-an-aerosol/particles/index.md)
- [Wall Loss Strategy System](wall-loss.md)
