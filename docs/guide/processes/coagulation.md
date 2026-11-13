# Coagulation Strategy System

## Complete direct-process context

The direct coagulation call is part of the illustrative [explicit-transfer sequence](https://github.com/Gorkowski/particula/blob/main/docs/examples/gpu/direct/gpu_complete_process_sequence.py), with [direct-process validation evidence](https://github.com/Gorkowski/particula/blob/main/particula/gpu/tests/process_sequence_test.py) and a [example regression](https://github.com/Gorkowski/particula/blob/main/examples_tests/gpu_complete_process_sequence_example_test.py). Persistent RNG remains caller-owned; the example is not a backend selector, scheduler, `Runnable`, resident loop, or CPU fallback.

> Strategy-based coagulation kernels with unified builders, factory selection, runnable integration, and particle-resolved support.

## Overview

The coagulation strategy system lets you model particle collisions using the same object-oriented patterns as condensation and wall loss. Instead of calling standalone kernel functions, you work with strategy objects that operate on native `ParticleData`, support multiple distribution types, and expose a consistent `kernel` / `rate` / `step` interface. Strategies cover Brownian motion, electrostatic enhancement, turbulence (shear and DNS), gravitational sedimentation, and user-defined combinations.

This feature is built around user-facing APIs exposed via `particula.dynamics`:

- `CoagulationStrategyABC` – abstract base class defining kernel, gain/loss/net rate, and `step` for discrete, continuous, or particle-resolved distributions.
- `BrownianCoagulationStrategy`, `ChargedCoagulationStrategy`, `TurbulentShearCoagulationStrategy`, `TurbulentDNSCoagulationStrategy`, `SedimentationCoagulationStrategy` – concrete physical models.
- `CombineCoagulationStrategy` – sums multiple kernels of the same distribution type for combined physics.
- Builders for each strategy plus mixins for distribution type, turbulent dissipation, fluid density, and charged-kernel validation.
- `CoagulationFactory` – factory for selecting a coagulation strategy by name with builder defaults.
- `Coagulation` runnable – delegates to a coagulation strategy, splits `time_step` across `sub_steps`, and composes with other runnables using the `|` operator.

The `Coagulation` runnable requires a native `Aerosol` holding `ParticleData`,
`GasData` and `EnvironmentData`. Each equal substep reads the current held
population and environment. Strategies must return the same particle container;
the runnable rejects replacement results. Gas state remains held independently.
The runnable's `rate` forwards the strategy's native rate without changing its
population measure. Non-native aggregates reject before strategy dispatch.

## Native kernel and rate contract (v0.3.0)

The six concrete CPU strategy families accept explicitly tagged `ParticleData`
for kernel evaluation: Brownian, charged, sedimentation, turbulent shear,
turbulent DNS and compatible combinations. Configuration stays on strategies.
Every native entry revalidates representation and physical state, requires one
box and matching `distribution_type`, and checks scalar temperature/pressure.
Mutated mechanism controls and combined child distributions are checked again.
Non-native objects and untagged populations reject at kernel, gain/loss/net-rate
and step boundaries; distribution meaning is never inferred. Combined kernels
validate the full population once, evaluate child material laws internally,
and restore zero interaction rows and columns for empty slots. This also
applies to nested combinations; private material storage is not a supported
population-execution input.

Kernel values have units m³/s and use per-particle material mass, geometry and
charge. They do not multiply physical mass by population count. Zero-mass,
zero-count slots retain zero rows/columns in the original slot layout; occupied
zero-mass particles reject. Existing CPU sedimentation and DNS retain their
arithmetic mean of per-particle mass-weighted effective density. DNS retains its
existing ambient-pressure convention; this migration does not change its law.

| Native quantity | Interpretation |
|---|---|
| Resolved concentration | Raw counts; density is N/V for positive physical volume |
| Discrete PMF concentration | Bin number density, m⁻³; volume must be exactly 1 m³ |
| Radius PDF concentration | dn/dr, m⁻⁴; integrate on explicit increasing `radius_grid`, at volume 1 m³ |
| Discrete gain/loss/net rate | Number-density change, m⁻³ s⁻¹ |
| Radius-PDF gain/loss/net rate | Radius-density change, m⁻⁴ s⁻¹ |
| Friction factor | kg/s |

PDF quadrature uses the declared radius coordinate; physical kernel radii are
still derived from material mass and density. PMF gain uses its existing
conversion to PDF on physical radii. The existing cubic gain interpolation
requires at least four increasing positive nodes for a nonzero population;
shared container metadata does not supply a lower-order alternative. Zero-work
rates remain zero. Supplied pair kernels must be real, finite, nonnegative
square matrices matching the population width.

Gain/loss/net-rate APIs remain discrete/PDF operations. Native resolved kernel
support does not introduce deterministic resolved rate APIs. The examples below
use native state. Independent native fixtures are in
`particula/dynamics/coagulation/coagulation_strategy/tests/native_rates_test.py`.

### Native CPU stepping and conservation

`step` requires explicitly tagged `ParticleData`, including on zero-duration
calls. Non-native objects and untagged populations reject before numerical
dispatch; the method never infers a distribution type. It preserves the input
container and its arrays. It validates
representation, writable/nonoverlapping destinations, environment and current
process controls before publishing a staged update. Rejected numerical
proposals leave caller arrays unchanged. Density, volume and the explicit PDF
radius grid are protected. These direct strategy methods remain single-box.
The `Coagulation` runnable stages and iterates all boxes, with explicit
`CPUExecutionContext` required for stochastic multi-box execution; see
[batched CPU execution](index.md#batched-cpu-runnable-execution).

Discrete/PDF stepping preserves the existing explicit number gain/loss
update on a fixed composition grid. It writes concentration only; per-bin
species masses and charge remain unchanged. Negative or nonfinite proposed
concentrations reject before publication, without clipping.

**Intended use:** this fixed-composition PMF/PDF method is suited to particles
with uniform composition across radius, or to a population represented as a
single effective material with a prescribed effective density. It is generally
not appropriate for tracking composition that varies with radius. An effective
density is a bulk material approximation, not a replacement for transporting
individual species between bins. Uniform composition does not remove the
method's finite-domain and discretization errors.

Use `particle_resolved` when species composition varies with radius or when
tracking more complex interactions that depend on individual particles'
composition, size, and charge. It retains per-particle species masses as
particles evolve and merge. See the
[native scientific process guide](index.md) for supported
process calls and constraints, including equal positive active counts for
resolved coagulation.

**Known limitation:** gained particles inherit the destination bin's fixed
composition rather than the combined donor composition. With composition
varying across particle size, this method does not generally conserve each
species or weighted charge. Refining the radius grid or reducing the timestep
does not resolve that composition error, even when total-mass error converges.
V0.3.0 deliberately preserves this method; conservative species/charge transport
is deferred and is not part of the supported density-step contract.

For an illustrative two-species, equal-density (1000 kg/m^3), constant-kernel
(1e-15 m^3/s) lognormal case, the PDF species defects at 1024 radius nodes are
approximately +7.79% and -12.20% **of gross transferred species mass**, while
total-mass error converges. The corresponding PMF defects are +7.43% and
-12.49%. These are fixture-specific characterization results, not universal
error bounds or percentages of the starting inventory. The fixture spans
1e-9 to 1e-6 m, has geometric mean 1e-7 m, log standard deviation .35 and
number density 1e10 m^-3; its first species fraction is
`clip(.5 + .3*log(r/1e-7), .05, .95)`. The independent refinement regression
retains both the nonzero species defects and the decreasing total-mass error.

Resolved stepping uses the existing binned/direct selector with equal positive
active counts within the box. Unequal active weights reject before drawing;
this selector's merge cannot conserve them. Collision scheduling includes
`common_count * time_step / volume`. Temporary kernel bins carry summed counts
and count-weighted material means at the original physical volume; authoritative
merges use original slot material and charge. Inactive slots do not participate.
The step stages per-species mass, count and charge, preserving weighted species
and charge inventories. Selection consumes the strategy's existing RNG;
rejection after draws does not promise RNG rollback or cross-backend seeded
equality.

Focused native stepping checks:

```bash
pytest particula/dynamics/coagulation/coagulation_strategy/tests/native_coagulation_step_test.py particula/dynamics/coagulation/coagulation_strategy/tests/native_coagulation_limitations_test.py -q --no-cov
```

## Key Benefits

- **Consistent dynamics workflow**: Same strategy-based API (`kernel`, `rate`, `step`, `distribution_type`) and runnable integration as condensation and wall loss.
- **Builder/factory parity with validation**: Fluent builders enforce distribution type, turbulent parameters, fluid density, and charged kernel strategy selection with optional unit conversion.
- **Coverage of major coagulation regimes**: Brownian, charged/electrostatic, turbulent shear, turbulent DNS, and sedimentation kernels plus additive combinations.
- **Distribution-type flexibility**: Works with "discrete", "continuous_pdf", and "particle_resolved" modes so you can reuse existing particle builders.
- **Pipeline ready**: `Coagulation` runnable uses `sub_steps`, rejects negative native density proposals, and chains with other runnables via `|` for end-to-end simulations.

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
import numpy as np
import particula as par

particles = par.particles.create_lognormal_binned_particle_data(
    distribution_type="discrete",
    radius_bins=np.geomspace(50e-9, 500e-9, 64),
    mode=[100e-9], geometric_standard_deviation=[1.5],
    number_concentration=[1e8],
)
gas = par.gas.GasData(
    name=["organic"], molar_mass=np.array([0.180]),
    concentration=np.array([[1e-6]]), partitioning=np.array([True]),
)
environment = par.gas.EnvironmentData(
    temperature=[298.15], pressure=[101325.0], saturation_ratio=[[1.0]],
)
aerosol = par.Aerosol(particles=particles, gas=gas, environment=environment)
coagulation = par.dynamics.Coagulation(
    coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
        distribution_type="discrete",
    ),
)

# Sub-steps split time_step internally
updated = coagulation.execute(
    aerosol,
    time_step=60.0,
    sub_steps=240,
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
updated = pipeline.execute(aerosol, time_step=60.0, sub_steps=240)
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
        "turbulent_dissipation": 1e-2,
        "turbulent_dissipation_units": "m^2/s^3",
        "fluid_density": 1.225,
        "fluid_density_units": "kg/m^3",
        "reynolds_lambda": 150.0,           # dimensionless
        "relative_velocity": 0.6,
        "relative_velocity_units": "m/s",
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
        par.dynamics.BrownianCoagulationStrategy(distribution_type="discrete"),
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
- "continuous_pdf" – radius number-density PDFs (`dn/dr` in m⁻⁴).
- "particle_resolved" – ensembles of individual particles with stochastic pair selection.

Select the appropriate mode at initialization (or via builders/factory). Particle-resolved paths use kernel-radius binning and count-weighted merges; discrete/continuous paths apply gain/loss rate updates.

### Particle-resolved coagulation

`CoagulationStrategyABC.step` handles particle-resolved updates by mapping to a kernel radius grid, drawing collision pairs, and updating via `collide_pairs`:

```python
resolved_particles = par.particles.create_resolved_particle_data(
    masses=[1e-18, 2e-18], density=1000.0,
    concentration=[1.0, 1.0], volume=1e-6,
)
particle_resolved = par.dynamics.BrownianCoagulationStrategy(
    distribution_type="particle_resolved",
)
resolved_particles = particle_resolved.step(
    particle=resolved_particles,
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
    .set_distribution_type("continuous_pdf")
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
    },
)
```

CPU snippets share
imports and the native aerosol constructed in the runnable section above;
optional GPU snippets only declare their separate low-level interfaces.
That runnable fixture uses an explicitly truncated 50–500 nm, 64-bin PMF
with a 100 nm mode, GSD 1.5 and 1e8 m⁻³ number density. Its finite interval is
an illustration, not a converged distribution or atmospheric calibration.
The 60/120-second runs use 0.25-second substeps to keep this fixture's native
proposals admissible. This does not establish a general timestep bound.

## Getting Started

### Quick start: Brownian coagulation on a discrete distribution

```python
import particula as par

# 1. Build a radius-binned particle distribution
particle = par.particles.create_lognormal_binned_particle_data(
    distribution_type="discrete",
)

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

- A native `ParticleData` instance (with an explicit population kind).
- Basic familiarity with particula dynamics and the runnable pipeline (`|`).

## Typical Workflows

### 1. Build native ParticleData

Use preset builders or custom particle-resolved constructors:

```python
particle = par.particles.create_lognormal_binned_particle_data(
    distribution_type="discrete", volume=1.0,
    radius_bins=np.geomspace(50e-9, 500e-9, 64),
    mode=[100e-9], geometric_standard_deviation=[1.5],
    number_concentration=[1e8],
)
aerosol.particles = particle
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
    sub_steps=480,
)
```

### 4. Compose with other dynamics in a single pipeline

Order processes explicitly using the `|` operator:

```python
from particula.dynamics.condensation.species_mapping import CondensationSpeciesMap

condensation = par.dynamics.MassCondensation(
    par.dynamics.CondensationIsothermal(
        molar_mass=gas.molar_mass.copy(),
        diffusion_coefficient=2e-5,
        accommodation_coefficient=1.0,
        activity_strategy=par.particles.ActivityIdealMass(),
        surface_strategy=par.particles.SurfaceStrategyVolume(density=1000.0),
        vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(1e-3),
        species_map=CondensationSpeciesMap(("organic",), ((0, 0),)),
    )
)
wall_loss = par.dynamics.WallLoss(
    wall_loss_strategy=par.dynamics.SphericalWallLossStrategy(
        wall_eddy_diffusivity=1e-3,
        chamber_radius=0.5,
        distribution_type="discrete",
    ),
)
pipeline = condensation | coagulation | wall_loss
updated = pipeline.execute(aerosol, time_step=60.0, sub_steps=240)
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
resolved_particles = par.particles.create_resolved_particle_data(
    masses=[1e-18, 2e-18], density=1000.0,
    concentration=[1.0, 1.0], volume=1e-6,
)
particle_resolved = par.dynamics.BrownianCoagulationStrategy(
    distribution_type="particle_resolved",
)
resolved_particles = particle_resolved.step(
    particle=resolved_particles,
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

1. **Match distribution type to particle builder:** Keep `distribution_type` consistent with how native `ParticleData` was constructed to avoid unintended kernel application.
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

- **Wall loss strategies**: [wall_loss_strategy_system.md](wall-loss.md)
- **Condensation strategies**: [condensation_strategy_system.md](condensation.md)
- **GPU direct example**: [gpu_coagulation_direct.py](https://github.com/Gorkowski/particula/blob/main/docs/examples/gpu/direct/gpu_coagulation_direct.py)
- **GPU roadmap**: [Pending initial-state optimization](../../contribute/roadmap.md)

## FAQ

### How do I choose between turbulent shear and turbulent DNS?

Use turbulent shear (`TurbulentShearCoagulationStrategy`) for Saffman–Turner regimes with moderate dissipation and when Reynolds lambda is not specified. Use turbulent DNS (`TurbulentDNSCoagulationStrategy`) when you have DNS-fit parameters (`reynolds_lambda`, `relative_velocity`) or want higher-Re effects.

### Can I mix charged and turbulent effects?

Yes. Build charged and turbulent strategies with the same `distribution_type`, then sum them via `CombineCoagulationStrategy` (or factory `strategy_type="combine"`) to add their kernels.

### How does particle-resolved stepping differ from discrete bins?

Particle-resolved mode bins a kernel radius grid, draws collision pairs stochastically, and calls `collide_pairs`; discrete/continuous modes apply gain/loss rates directly to concentrations.

## See Also

- [Wall Loss Strategy System](wall-loss.md)
- [Condensation Strategy System](condensation.md)
- [Dynamics overview](../../index.md)
