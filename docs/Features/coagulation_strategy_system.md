# Coagulation Strategy System

> Strategy-based coagulation kernels with unified builders, factory selection, runnable integration, and particle-resolved support.

## Overview

The coagulation strategy system lets you model particle collisions using the same object-oriented patterns as condensation and wall loss. Instead of calling standalone kernel functions, you work with strategy objects that operate on `ParticleRepresentation`, support multiple distribution types, and expose a consistent `kernel` / `rate` / `step` interface. Strategies cover Brownian motion, electrostatic enhancement, turbulence (shear and DNS), gravitational sedimentation, and user-defined combinations.

This feature is built around user-facing APIs exposed via `particula.dynamics`:

- `CoagulationStrategyABC` – abstract base class defining kernel, gain/loss/net rate, and `step` for discrete, continuous, or particle-resolved distributions.
- `BrownianCoagulationStrategy`, `ChargedCoagulationStrategy`, `TurbulentShearCoagulationStrategy`, `TurbulentDNSCoagulationStrategy`, `SedimentationCoagulationStrategy` – concrete physical models.
- `CombineCoagulationStrategy` – sums multiple kernels of the same distribution type for combined physics.
- Builders for each strategy plus mixins for distribution type, turbulent dissipation, fluid density, and charged-kernel validation.
- `CoagulationFactory` – factory for selecting a coagulation strategy by name with builder defaults.
- `Coagulation` runnable – delegates to a coagulation strategy, splits `time_step` across `sub_steps`, and composes with other runnables using the `|` operator.

## Key Benefits

- **Consistent dynamics workflow**: Same strategy-based API (`kernel`, `rate`, `step`, `distribution_type`) and runnable integration as condensation and wall loss.
- **Builder/factory parity with validation**: Fluent builders enforce distribution type, turbulent parameters, fluid density, and charged kernel strategy selection with optional unit conversion.
- **Coverage of major coagulation regimes**: Brownian, charged/electrostatic, turbulent shear, turbulent DNS, and sedimentation kernels plus additive combinations.
- **Distribution-type flexibility**: Works with "discrete", "continuous_pdf", and "particle_resolved" modes so you can reuse existing particle builders.
- **Pipeline ready**: `Coagulation` runnable uses `sub_steps`, clamps through strategy logic, and chains with other runnables via `|` for end-to-end simulations.

## Who It's For

This feature is designed for:

- **Chamber and box-model users**: Running time-dependent simulations where coagulation competes with condensation, wall loss, dilution, or emissions.
- **Particle-resolved modelers**: Requiring discrete collision pairs, kernel radius binning, and stochastic pair selection.
- **Turbulence and charged-aerosol researchers**: Exploring electrostatic enhancement or turbulence-driven collision rates without re-implementing kernels.
- **Model developers**: Extending with new kernels while reusing validation, distribution handling, and runnable plumbing.

## Capabilities

### Unified coagulation API in `particula.dynamics`

Coagulation is exposed alongside other dynamics components:

```python
import particula as par

# Abstract interface
par.dynamics.CoagulationStrategyABC

# Concrete strategies
par.dynamics.BrownianCoagulationStrategy
par.dynamics.ChargedCoagulationStrategy
par.dynamics.TurbulentShearCoagulationStrategy
par.dynamics.TurbulentDNSCoagulationStrategy
par.dynamics.SedimentationCoagulationStrategy

# Combination and factory
par.dynamics.CombineCoagulationStrategy
par.dynamics.CoagulationFactory
```

All strategies share a common shape:

- Initialize with physical parameters (e.g., turbulent dissipation, fluid density, charged kernel strategy) and `distribution_type`.
- Call `kernel(particle, temperature, pressure)` to compute the dimensional kernel [m^3/s].
- Call `net_rate(...)`, `loss_rate(...)`, or `gain_rate(...)` as needed.
- Call `step(particle, temperature, pressure, time_step)` to advance the system.

### Runnable entry point: `Coagulation`

`Coagulation` is a `RunnableABC` exported as `par.dynamics.Coagulation`. It operates on an `Aerosol`, delegates `step` to the provided coagulation strategy, splits `time_step` across any `sub_steps`, and composes with other runnables using `|`:

```python
import particula as par

aerosol = ...  # your Aerosol instance
coagulation = par.dynamics.Coagulation(
    coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
        distribution_type="discrete",
    ),
)

# Sub-steps split time_step internally
updated = coagulation.execute(
    aerosol,
    time_step=60.0,
    sub_steps=4,
)

# Chain with wall loss or condensation in one pipeline
wall_loss = par.dynamics.WallLoss(
    wall_loss_strategy=par.dynamics.SphericalWallLossStrategy(
        wall_eddy_diffusivity=1e-3,
        chamber_radius=0.5,
        distribution_type="discrete",
    ),
)
pipeline = coagulation | wall_loss
updated = pipeline.execute(aerosol, time_step=60.0)
```

### Brownian coagulation strategy

`BrownianCoagulationStrategy` implements the classical Brownian kernel using `get_brownian_kernel_via_system_state`. It works across discrete, continuous, and particle-resolved distributions.

Typical initialization:

```python
brownian = par.dynamics.BrownianCoagulationStrategy(
    distribution_type="discrete",
)
```

### Charged coagulation with kernel strategies

`ChargedCoagulationStrategy` wraps electrostatic kernel strategies (e.g., `HardSphereKernelStrategy`, `Coulomb*` variants). Provide a `ChargedKernelStrategyABC` instance:

```python
charged_kernel = par.dynamics.HardSphereKernelStrategy()
charged = par.dynamics.ChargedCoagulationStrategy(
    distribution_type="discrete",
    kernel_strategy=charged_kernel,
)
```

Use the builder for validation and readability:

```python
charged = (
    par.dynamics.ChargedCoagulationBuilder()
    .set_distribution_type("discrete")
    .set_charged_kernel_strategy(par.dynamics.HardSphereKernelStrategy())
    .build()
)
```

### Turbulent shear coagulation

`TurbulentShearCoagulationStrategy` uses the Saffman–Turner (1956) shear kernel. Requires turbulent dissipation [m^2/s^3] and fluid density [kg/m^3]:

```python
turb_shear = par.dynamics.TurbulentShearCoagulationStrategy(
    distribution_type="continuous_pdf",
    turbulent_dissipation=1e-3,
    fluid_density=1.225,
)
```

Builder example with unit conversion and validation:

```python
turb_shear = (
    par.dynamics.TurbulentShearCoagulationBuilder()
    .set_distribution_type("continuous_pdf")
    .set_turbulent_dissipation(1e-3, "m^2/s^3")
    .set_fluid_density(1.225, "kg/m^3")
    .build()
)
```

### Turbulent DNS coagulation

`TurbulentDNSCoagulationStrategy` follows Ayala et al. (2008) DNS fits. Requires turbulent dissipation, fluid density, Reynolds lambda (dimensionless), and relative velocity [m/s]:

```python
turb_dns = par.dynamics.TurbulentDNSCoagulationStrategy(
    distribution_type="discrete",
    turbulent_dissipation=0.01,
    fluid_density=1.225,
    reynolds_lambda=74.0,
    relative_velocity=0.5,
)
```

Factory configuration dict example (via `CoagulationFactory`):

```python
factory = par.dynamics.CoagulationFactory()
turb_dns = factory.get_strategy(
    strategy_type="turbulent_dns",
    parameters={
        "distribution_type": "discrete",
        "turbulent_dissipation": 1e-2,      # m^2/s^3
        "fluid_density": 1.225,             # kg/m^3
        "reynolds_lambda": 150.0,           # dimensionless
        "relative_velocity": 0.6,           # m/s
    },
)
```

### Sedimentation coagulation

`SedimentationCoagulationStrategy` models gravitational-settling-driven collisions (Seinfeld & Pandis, 2016) and works across supported distribution types. No extra parameters beyond `distribution_type` are required.

### Combined coagulation strategies

`CombineCoagulationStrategy` sums kernels from multiple strategies (distribution type must match). Example: Brownian + turbulent shear:

```python
combined = par.dynamics.CombineCoagulationStrategy(
    strategies=[
        par.dynamics.BrownianCoagulationStrategy("discrete"),
        par.dynamics.TurbulentShearCoagulationStrategy(
            distribution_type="discrete",
            turbulent_dissipation=1e-3,
            fluid_density=1.225,
        ),
    ]
)
```

### Support for multiple distribution types

The strategy system operates on the same distribution types used elsewhere in particula:

- "discrete" – radius-binned distributions.
- "continuous_pdf" – continuous probability-density representations.
- "particle_resolved" – ensembles of individual particles with stochastic pair selection.

Select the appropriate mode at initialization (or via builders/factory). Particle-resolved paths use kernel-radius binning and `collide_pairs` updates; discrete/continuous paths apply gain/loss rate updates.

### Particle-resolved coagulation

`CoagulationStrategyABC.step` handles particle-resolved updates by mapping to a kernel radius grid, drawing collision pairs, and updating via `collide_pairs`:

```python
particle_resolved = par.dynamics.BrownianCoagulationStrategy(
    distribution_type="particle_resolved",
)
aerosol.particles = particle_resolved.step(
    particle=aerosol.particles,
    temperature=298.15,
    pressure=101325.0,
    time_step=1.0,
)
```

### Builder and factory workflow

Builders provide validated, unit-aware construction; the factory selects a strategy by name using those builders under the hood:

```python
import particula as par

# Brownian via builder
brownian = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("discrete")
    .build()
)

# Charged via builder with kernel strategy
charged = (
    par.dynamics.ChargedCoagulationBuilder()
    .set_distribution_type("continuous_pdf")
    .set_charged_kernel_strategy(par.dynamics.HardSphereKernelStrategy())
    .build()
)

# Factory selection (combine)
factory = par.dynamics.CoagulationFactory()
combined = factory.get_strategy(
    strategy_type="combine",
    parameters={
        "strategies": [brownian, charged],
        "distribution_type": "continuous_pdf",
    },
)
```

## Getting Started

### Quick start: Brownian coagulation on a discrete distribution

```python
import particula as par

# 1. Build a radius-binned particle distribution
particle = par.particles.PresetParticleRadiusBuilder().build()

# 2. Configure Brownian coagulation
brownian = par.dynamics.BrownianCoagulationStrategy(
    distribution_type="discrete",
)

# 3. Compute instantaneous kernel and rates
T = 298.15  # K
P = 101325.0  # Pa
kernel = brownian.kernel(particle, temperature=T, pressure=P)
net = brownian.net_rate(particle, temperature=T, pressure=P)

# 4. Advance by one time step
particle = brownian.step(
    particle=particle,
    temperature=T,
    pressure=P,
    time_step=10.0,  # s
)
```

### Prerequisites

- `particula` version 0.2.6 or later installed.
- A `ParticleRepresentation` instance (e.g., from preset particle builders).
- Basic familiarity with particula dynamics and the runnable pipeline (`|`).

## Typical Workflows

### 1. Build a ParticleRepresentation

Use preset builders or custom particle-resolved constructors:

```python
particle = (
    par.particles.PresetParticleRadiusBuilder()
    .set_volume(1.0, "m^3")
    .build()
)
```

### 2. Configure a coagulation strategy

Pick a physical model and distribution type:

```python
coag_strategy = par.dynamics.TurbulentShearCoagulationStrategy(
    distribution_type="discrete",
    turbulent_dissipation=5e-4,
    fluid_density=1.2,
)
```

### 3. Run through the `Coagulation` runnable (with `sub_steps`)

```python
coagulation = par.dynamics.Coagulation(coagulation_strategy=coag_strategy)
updated = coagulation.execute(
    aerosol,
    time_step=120.0,
    sub_steps=3,
)
```

### 4. Compose with other dynamics in a single pipeline

Order processes explicitly using the `|` operator:

```python
condensation = par.dynamics.MassCondensation(
    condensation_strategy=par.dynamics.CondensationIsothermal(
        molar_mass=0.18,
        diffusion_coefficient=2e-5,
        accommodation_coefficient=1.0,
    ),
)
wall_loss = par.dynamics.WallLoss(
    wall_loss_strategy=par.dynamics.SphericalWallLossStrategy(
        wall_eddy_diffusivity=1e-3,
        chamber_radius=0.5,
        distribution_type="discrete",
    ),
)
pipeline = condensation | coagulation | wall_loss
updated = pipeline.execute(aerosol, time_step=60.0, sub_steps=2)
```

### 5. Builder/factory workflow for reproducibility

```python
factory = par.dynamics.CoagulationFactory()
coag_strategy = factory.get_strategy(
    strategy_type="brownian",
    parameters={"distribution_type": "continuous_pdf"},
)
coagulation = par.dynamics.Coagulation(coagulation_strategy=coag_strategy)
```

### 6. Particle-resolved collision loop

```python
particle_resolved = par.dynamics.BrownianCoagulationStrategy(
    distribution_type="particle_resolved",
)
aerosol.particles = particle_resolved.step(
    particle=aerosol.particles,
    temperature=300.0,
    pressure=101325.0,
    time_step=0.5,
)
```

## Use Cases

### Use case 1: Brownian-only chamber decay

**Scenario:** You need size-dependent Brownian coagulation over a 10-minute chamber experiment.

**Solution:** Use `BrownianCoagulationStrategy` with `Coagulation` runnable and `sub_steps` to maintain stability at small `time_step` slices.

### Use case 2: Charged aerosol aging

**Scenario:** You are studying charge-enhanced coagulation (e.g., ion-induced charging or corona discharge) and want to compare kernel strategies.

**Solution:** Instantiate `ChargedCoagulationStrategy` with `HardSphereKernelStrategy` (or other Coulomb models) and sweep voltages/charges externally while keeping the same runnable pipeline.

### Use case 3: Turbulent industrial flow

**Scenario:** You want turbulence-driven coagulation rates for an industrial duct with known dissipation and Reynolds lambda.

**Solution:** Configure `TurbulentDNSCoagulationStrategy` via `CoagulationFactory` using measured `turbulent_dissipation`, `fluid_density`, `reynolds_lambda`, and `relative_velocity`; run through `Coagulation` with other dynamics chained.

## Configuration

| Option | Description | Default / Units | Applies To |
|--------|-------------|-----------------|------------|
| `distribution_type` | "discrete", "continuous_pdf", or "particle_resolved" | Required | All strategies & builders |
| `charged_kernel_strategy` | Electrostatic kernel (e.g., `HardSphereKernelStrategy`) | Required | Charged |
| `turbulent_dissipation` | Turbulent energy dissipation rate | Required, m^2/s^3 | Turbulent shear, Turbulent DNS |
| `fluid_density` | Fluid density for the medium | Required, kg/m^3 | Turbulent shear, Turbulent DNS |
| `reynolds_lambda` | Taylor-scale Reynolds number | Required | Turbulent DNS |
| `relative_velocity` | Relative particle-flow velocity | Required, m/s | Turbulent DNS |
| `strategies` | List of strategies to sum | Required | Combine |
| `particle_resolved_kernel_radius` | Kernel-radius grid (optional) | Optional | Particle-resolved modes |
| `particle_resolved_kernel_bins_number` | Number of kernel bins | Optional | Particle-resolved modes |
| `particle_resolved_kernel_bins_per_decade` | Bins per radius decade | `10` | Particle-resolved modes |

## Best Practices

1. **Match distribution type to particle builder:** Keep `distribution_type` consistent with how `ParticleRepresentation` was constructed to avoid unintended kernel application.
2. **Use builders/factory for validation:** Let builders enforce turbulence units, fluid density, and charged-kernel selection; avoid manual instantiation when sharing configurations.
3. **Start with smaller `time_step` and use `sub_steps`:** Especially for particle-resolved or highly turbulent cases to reduce overshoot in collisions.
4. **Combine kernels deliberately:** Use `CombineCoagulationStrategy` when you need additive physics (e.g., Brownian + turbulent shear); ensure all strategies share the same distribution type.
5. **Keep parameters physical:** Validate dissipation rates, Reynolds lambda, and relative velocities against your experiment or CFD results before running long simulations.

## Limitations

- Dimensionless kernels are not implemented for combined, turbulent, or sedimentation strategies; use dimensional forms via `kernel`.
- All strategies in `CombineCoagulationStrategy` must share the same `distribution_type`.
- No built-in high-level orchestrator; you control the time loop and pipeline ordering.
- DNS parameterization assumes applicability of Ayala et al. (2008) fits; verify for extreme regimes.

## Related Documentation

- **Testing guide**: [adw-docs/testing_guide.md](../../adw-docs/testing_guide.md)
- **Code style**: [adw-docs/code_style.md](../../adw-docs/code_style.md)
- **Architecture reference**: [adw-docs/architecture_reference.md](../../adw-docs/architecture_reference.md)
- **Wall loss strategies**: [wall_loss_strategy_system.md](./wall_loss_strategy_system.md)
- **Condensation strategies**: [condensation_strategy_system.md](./condensation_strategy_system.md)

## FAQ

### How do I choose between turbulent shear and turbulent DNS?

Use turbulent shear (`TurbulentShearCoagulationStrategy`) for Saffman–Turner regimes with moderate dissipation and when Reynolds lambda is not specified. Use turbulent DNS (`TurbulentDNSCoagulationStrategy`) when you have DNS-fit parameters (`reynolds_lambda`, `relative_velocity`) or want higher-Re effects.

### Can I mix charged and turbulent effects?

Yes. Build charged and turbulent strategies with the same `distribution_type`, then sum them via `CombineCoagulationStrategy` (or factory `strategy_type="combine"`) to add their kernels.

### How does particle-resolved stepping differ from discrete bins?

Particle-resolved mode bins a kernel radius grid, draws collision pairs stochastically, and calls `collide_pairs`; discrete/continuous modes apply gain/loss rates directly to concentrations.

## See Also

- [Wall Loss Strategy System](./wall_loss_strategy_system.md)
- [Condensation Strategy System](./condensation_strategy_system.md)
- [Dynamics overview](../index.md)
