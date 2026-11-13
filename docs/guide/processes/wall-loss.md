# Wall Loss Strategy System

## Complete direct-process context

The direct wall-loss call appears in the illustrative [explicit-transfer sequence](https://github.com/Gorkowski/particula/blob/main/docs/examples/gpu/direct/gpu_complete_process_sequence.py), with [direct-process validation evidence](https://github.com/Gorkowski/particula/blob/main/particula/gpu/tests/process_sequence_test.py) and a [example regression](https://github.com/Gorkowski/particula/blob/main/examples_tests/gpu_complete_process_sequence_example_test.py). Caller-owned persistent RNG and sidecars remain explicit; this is not a `Runnable`, backend selector, scheduler, or CPU fallback.

> Strategy-based wall loss for chamber simulations that plugs directly into particula's dynamics workflow.

## Overview

The wall loss strategy system lets you model particle deposition onto chamber walls using the same object-oriented patterns as condensation and coagulation. Instead of calling standalone rate functions, you work with strategy objects that operate on native `ParticleData`, support multiple distribution types, and expose a consistent `rate` / `step` interface.

This feature is built around user-facing APIs exposed via `particula.dynamics`:

- `WallLossStrategy` – abstract base class for wall loss models.
- `SphericalWallLossStrategy`, `RectangularWallLossStrategy`, and
  `ChargedWallLossStrategy` – chamber implementations using existing wall loss
  coefficient utilities plus electrostatic modifiers for charged particles.
- `SphericalWallLossBuilder`, `RectangularWallLossBuilder`, and
  `ChargedWallLossBuilder` – validated, unit-aware builders for configuring
  strategies and electrostatic parameters (wall potential, electric field).
- `WallLossFactory` – factory for selecting a wall loss geometry by name with
  builder defaults.
- `WallLoss` runnable – delegates to a wall loss strategy on `Aerosol`, splits
  `time_step` across `sub_steps`, and delegates native positivity validation to the strategy.

## Native CPU strategy contract (v0.3.0)

The neutral spherical/rectangular and charged spherical/rectangular strategies
accept explicitly tagged, single-box `ParticleData` directly. Configuration
remains on the strategy. Each coefficient, rate and step call revalidates the
current representation, physical state, environment, geometry and field values.
Strategy and particle distribution types must agree. Facade execution has
been removed; population-based entry points require native `ParticleData`.
Native runnables retain the held containers; see the composed examples below.

Charged coefficients read the current particle charge arrays. For independent
array calculations, use `compute_coefficient_from_arrays` with explicit charge.
The charge-free `loss_coefficient_for_particles` interface assumes neutral
particles and does not inherit charge from earlier calls.

| Representation | Stored population | Rate measure | Native step |
|---|---|---|---|
| Discrete PMF | Bin number density, m⁻³, at V=1 | m⁻³ s⁻¹ | Existing forward Euler loss |
| Radius PDF | dn/dr, m⁻⁴, at V=1 | m⁻⁴ s⁻¹ | Existing forward Euler loss; integrate inventories on `radius_grid` |
| Particle resolved | Counts at positive physical V | Count/V per second, m⁻³ s⁻¹ | Bernoulli survival of each weighted representative |

Density steps preserve per-particle mass and charge and publish only the
concentration update. A negative or nonfinite Euler proposal rejects before
mutation; use a smaller timestep. No exponential deterministic solver or
concentration clipping is introduced. The coefficient laws retain their
existing neutral transport, image-charge enhancement at zero potential, signed
field drift and coefficient limiting behavior.

Resolved survival uses `p = exp(-k * time_step)`. Unequal positive counts are
supported: a representative either retains its entire raw count or is removed.
Changing V changes density accounting, not its per-particle survival hazard.
The existing neutral CPU helper draws for positive-radius slots, including
unoccupied material carriers; charged CPU removal draws for occupied
positive-radius slots. Native zero-time and wholly unoccupied calls perform
validation but draw nothing. Rejections after a draw do not roll back RNG state.

Removed occupied slots have their counts, species masses and charge cleared,
so removed slots satisfy the fixed-slot activation contract. Surviving and
untouched inactive storage is preserved; the operation does not repair
pre-existing inactive material. The current approach clears removed slots without changing the CPU Bernoulli law.
GPU removal also clears charge, with its own RNG lifecycle.

Steps preserve the container and primary array identities. Preflight rejects
unwritable destinations and aliases with protected state or array-backed
geometry/field configuration. Density, volume and the PDF radius grid remain protected. Binned calls also
protect masses and charge; resolved removal additionally writes both. Independent tests
account for removed species and weighted charge, rather than requiring a sink
process to conserve its remaining inventory.

Both CPU strategy suites must be run explicitly, in separate invocations:

```bash
pytest particula/dynamics/tests/wall_loss_strategies_test.py -q --no-cov
pytest particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py -q --no-cov
pytest particula/dynamics/tests/native_wall_loss_test.py -q --no-cov
```

The adjacent `wall_loss/tests` directory is excluded from recursive discovery;
the full package runner cannot substitute for its explicit check.

## Key Benefits

- **Consistent dynamics workflow**: Use the same strategy-based API (`rate`, `step`, `distribution_type`) as for condensation and coagulation.
- **Builder/factory parity with validation**: Configure wall loss using the same unit-aware builder and factory patterns as other dynamics modules, with built-in checks for geometry and distribution types.
- **Direct integration with ParticleData**: Apply wall loss directly to particle distributions without writing glue code.
- **Extensible wall loss models**: Add new chamber geometries or wall loss models as additional `WallLossStrategy` implementations.

## Who It's For

This feature is designed for:

- **Chamber simulation users**: Running time-dependent simulations where wall deposition competes with processes like condensation, coagulation, and dilution.
- **Model developers**: Implementing new wall loss parameterizations while reusing particula's particle and dynamics infrastructure.
- **Experiment interpreters**: Matching chamber experiments with simulations that treat wall loss as a first-class process.

## Capabilities

### Unified wall loss API in `particula.dynamics`

Wall loss is exposed alongside other dynamics components:

```python
import particula as par

# Abstract interface (not instantiated directly)
par.dynamics.WallLossStrategy

# Concrete chamber implementations
par.dynamics.SphericalWallLossStrategy
par.dynamics.RectangularWallLossStrategy

# Builder and factory API (also re-exported via particula.dynamics.wall_loss)
par.dynamics.SphericalWallLossBuilder
par.dynamics.RectangularWallLossBuilder
par.dynamics.WallLossFactory
```

All wall loss strategies share a common shape:

- Initialize with physical parameters (e.g., wall eddy diffusivity, chamber
  radius).
- Call `rate(particle, temperature, pressure)` to compute instantaneous loss
  rate.
- Call `step(particle, temperature, pressure, time_step)` to advance the
  system.

#### Charged wall loss support

`ChargedWallLossStrategy` augments neutral wall loss with image-charge and
optional electric-field drift effects. Configure it via
`ChargedWallLossBuilder` or directly:

```python
charged_wall_loss = par.dynamics.ChargedWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_geometry="spherical",
    chamber_radius=0.5,
    wall_potential=1.0,        # volts (optional)
    wall_electric_field=25.0,  # V/m (optional drift term)
    distribution_type="discrete",
)
```

### Runnable entry point: `WallLoss`

`WallLoss` is a `RunnableABC` implementation exported as `par.dynamics.WallLoss`.
It operates on an `Aerosol` in the runnable pipeline, delegates `rate` and
`step` to the provided wall loss strategy, splits `time_step` across any
`sub_steps`, rejects negative native Euler proposals, and
works with spherical or rectangular strategies across all supported
distribution types.

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

wall_loss_strategy = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_radius=0.5,
    distribution_type="discrete",
)
wall_loss = par.dynamics.WallLoss(
    wall_loss_strategy=wall_loss_strategy,
)

# Sub-steps split time_step; native negative proposals reject.
aerosol = wall_loss.execute(
    aerosol,
    time_step=60.0,
    sub_steps=240,
)

# Chain with another runnable in a single pipeline
coagulation = par.dynamics.Coagulation(
    coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
        distribution_type="discrete",
    ),
)
combined = coagulation | wall_loss
aerosol = combined.execute(aerosol, time_step=60.0, sub_steps=240)
```

## Direct GPU charged wall loss

The GPU boundary is a low-level, caller-managed fixed-capacity slot operation. It is
not an extension of `WallLoss`, does not create a runnable, and does not
transfer data between CPU and Warp. Import only the direct step from the kernel
package; the configuration intentionally remains a concrete-module import:

```python
from particula.gpu.kernels import wall_loss_step_gpu
from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

config = NeutralWallLossConfig(
    geometry="spherical",
    wall_eddy_diffusivity=1e-3,  # m^2/s
    chamber_radius=0.5,  # m
    mode="charged",
    wall_potential=-1.0,  # V
    wall_electric_field=25.0,  # V/m; signed spherical scalar
)
```

The direct boundary supports only `particle_resolved` storage. Its SI inputs
are wall eddy diffusivity [m^2/s], chamber radius or dimensions [m],
temperature [K], pressure [Pa], time [s], signed wall potential [V], and
electric field [V/m]. A spherical configuration requires a positive radius and
no dimensions. A rectangular configuration requires exactly three positive
dimensions, no radius, and a caller-owned, same-device `wp.float64` electric
field vector shaped `(3,)`.

For charged spherical execution, the signed scalar field is preserved and the
signed potential-derived field is added. For charged rectangular execution,
the three field components are reduced to their Euclidean magnitude before the
signed potential-derived field is added; individual component signs do not
select a drift direction. Nonzero-charge slots use image-charge enhancement
and field drift, including image-charge enhancement when `wall_potential=0`.
Charged zero-charge slots use the exact neutral coefficient and RNG path.

Callers own `WarpParticleData.charge`, device placement, explicit CPU-to-Warp
transfer, synchronization, and any optional persistent `(n_boxes,)` `wp.uint32`
RNG sidecar. Successful positive-time calls mutate the supplied fixed-capacity
particle container in place and return that same container. Selected removed
slots have every mass lane, concentration, and charge cleared; surviving,
unselected storage retains its capacity and other fields. Rejected preflight
calls do not mutate caller-owned particle or RNG state. Rollback is not
promised after a mutation kernel launches.

Native tagged inputs use resolved raw counts, including unequal positive
weights, with positive physical box volumes. The removal probability is
`1 - exp(-k * time_step)` for each eligible representative; surviving counts
are unchanged. Removed species inventory is `sum(mass * count)` and its
concentration is that inventory divided by box volume. No volume factor is
introduced into the per-particle wall-loss coefficient. Native preflight
rejects occupied zero-mass slots, unrepresentable derived geometry/count
density, noncontiguous storage, and overlapping writers or protected inputs.
Explicit resolved population metadata is required at direct, prepared, and
selected-box boundaries. Untagged and PMF/PDF inputs reject before payload
access, including zero-duration calls.

The private prepared boundary freezes array references and scalar arguments.
Its enqueue performs no host setup or transfer, but the frozen kernel's
`initialize_rng=True` flag still requests a reset on each eligible launch.
For persistent replay, supply initialized caller-owned RNG state and prepare
with `initialize_rng=False`. Zero-time and wholly inactive calls do not advance
or reset supplied words. Native resident dispatch uses explicit resolved
metadata and session-owned continuation streams; see the
[resident walkthrough](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/resident/gpu_resident_multi_timestep.py). Direct
caller-owned RNG and resident lifecycle ownership are distinct boundaries.

Warp CPU is the baseline when Warp is installed. CUDA is optional and the
focused suites skip cleanly when it is unavailable:

```bash
pytest particula/gpu/dynamics/tests/wall_loss_funcs_test.py -q -Werror
pytest particula/gpu/kernels/tests/wall_loss_test.py \
  particula/gpu/kernels/tests/wall_loss_parity_test.py \
  particula/gpu/kernels/tests/wall_loss_native_test.py -q -Werror
```

Hidden transfer or CPU fallback, dynamic-slot management, scheduler/runnable or
backend integration, graph capture, differentiability, exact cross-backend RNG
replay, mandatory CUDA support, and performance claims remain deferred.

### Spherical wall loss strategy

`SphericalWallLossStrategy` models deposition in a well-mixed spherical chamber. It:

- Reuses existing wall loss coefficient utilities for the underlying physics.
- Works with single-box native `ParticleData` whose declared kind matches the strategy.
- Computes size-dependent loss rates and updates particle concentration over time.

Typical initialization:

```python
wall_loss = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=1e-3,  # m^2/s
    chamber_radius=0.5,          # m
    distribution_type="discrete",
)
```

### Rectangular wall loss strategy

`RectangularWallLossStrategy` models deposition in box-shaped chambers. Pair it with `RectangularWallLossBuilder` to validate a (length, width, height) tuple and convert units before constructing the strategy.

### Builder and factory workflow

Builders and the factory give you a validated, unit-aware way to construct wall loss strategies and keep parity with other dynamics modules. Geometry lengths convert to meters and wall eddy diffusivity converts to 1/s; setters validate positivity and enforce a length-3 tuple for rectangular chambers. Distribution types are restricted to the supported set and default to `"discrete"`.

```python
import particula as par

# Chained builder with unit conversion and validation
wall_loss = (
    par.dynamics.SphericalWallLossBuilder()
    .set_wall_eddy_diffusivity(1e-3, "1/s")
    .set_chamber_radius(50.0, "cm")  # converts to 0.5 m
    .set_distribution_type("discrete")
    .build()
)
```

For rectangular chambers, use `set_chamber_dimensions((L, W, H), units)`; each side must be positive and provided as a 3-tuple.

```python
factory = par.dynamics.WallLossFactory()
rectangular = factory.get_strategy(
    strategy_type="rectangular",
    parameters={
        "wall_eddy_diffusivity": 1e-4,
        "chamber_dimensions": (1.0, 0.5, 0.5),
        "distribution_type": "continuous_pdf",
    },
)
```

`WallLossFactory` is exported via both `particula.dynamics.wall_loss` and `particula.dynamics`, letting you select a geometry by name without manually instantiating builders.

### Support for multiple distribution types

The strategy system operates on the same distribution types used elsewhere in particula:

- "discrete" – radius-binned distributions.
- "continuous_pdf" – radius number-density PDFs (`dn/dr` in m⁻⁴).
- "particle_resolved" – ensembles of individual particles.

You select the appropriate mode at initialization (or via builder/factory parameters) with `distribution_type`, and the strategy adjusts its `step` behavior to match the representation.


The snippets on this page use the v0.3.0 development checkout. CPU snippets share
imports and the native aerosol constructed in the runnable section above;
optional GPU snippets only declare their separate low-level interfaces.
That runnable fixture uses an explicitly truncated 50–500 nm, 64-bin PMF
with a 100 nm mode, GSD 1.5 and 1e8 m⁻³ number density. Its finite interval is
an illustration, not a converged distribution or atmospheric calibration.
The 60/120-second runs use 0.25-second substeps to keep this fixture's native
proposals admissible. This does not establish a general timestep bound.

## Getting Started

### Quick start: wall loss on a discrete distribution

```python
import particula as par

# 1. Build a radius-binned particle distribution
particle = par.particles.create_lognormal_binned_particle_data(
    distribution_type="discrete",
)

# 2. Configure spherical wall loss
wall_loss = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_radius=0.5,
    distribution_type="discrete",
)

# 3. Compute instantaneous wall loss rate
T = 298.15  # K
P = 101325.0  # Pa
rate = wall_loss.rate(particle, temperature=T, pressure=P)

# 4. Advance 10 seconds with 40 equal steps (0.25 s each).
# Native binned wall loss rejects a step that would make a bin negative.
for _ in range(40):
    particle = wall_loss.step(
        particle=particle, temperature=T, pressure=P, time_step=0.25,
    )
```

### Prerequisites

- The v0.3.0 development checkout installed with `pip install -e .`; no older-wheel compatibility is implied.
- A native `ParticleData` instance (with an explicit population kind).
- Basic familiarity with particula's dynamics and particle-phase examples.

## Typical Workflows

### 1. Build native ParticleData

Start by constructing a particle distribution using existing builders:

```python
particle = par.particles.create_lognormal_binned_particle_data(
    distribution_type="discrete", volume=1.0,
    radius_bins=np.geomspace(50e-9, 500e-9, 64),
    mode=[100e-9], geometric_standard_deviation=[1.5],
    number_concentration=[1e8],
)
aerosol.particles = particle
```

Use the native resolved initializer for counts at a physical volume, or
`distribution_type="continuous_pdf"` with a radius grid for a radius PDF.
The strategy kind must match the stored kind; CPU execution is single-box.

### 2. Configure a SphericalWallLossStrategy

Choose physical parameters for your chamber and distribution type:

```python
wall_loss = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_radius=0.5,
    distribution_type="discrete",  # or "continuous_pdf" / "particle_resolved"
)
```

At this point you can:

- Inspect `wall_loss.rate(...)` to understand size-dependent loss.
- Call `wall_loss.step(...)` in a loop to model concentration decay.

### 3. Combine wall loss with other dynamics

Wall loss strategies are designed to compose with other dynamics, such as condensation and coagulation, in a single time-stepping loop:

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
wall = par.dynamics.WallLoss(wall_loss_strategy=wall_loss)
workflow = condensation | wall
for _ in range(10):
    aerosol = workflow.execute(aerosol, time_step=1.0, sub_steps=2)
```

This pattern matches how other dynamics strategies are combined in chamber simulations.

## Use Cases

### Use case 1: Standalone chamber wall loss

**Scenario:** You want to understand how quickly particles are lost to the walls of a well-mixed spherical chamber.

**Solution:** Build a radius-binned native `ParticleData`, configure `SphericalWallLossStrategy`, and integrate `step` over the experiment duration to track normalized concentration decay.

### Use case 2: Full chamber dynamics with wall loss

**Scenario:** You are modeling a chamber experiment where condensation growth, coagulation, and wall loss all act simultaneously.

**Solution:** Combine `SphericalWallLossStrategy` with existing condensation or coagulation strategies in a shared time loop. This lets you track how wall loss interacts with growth and collisions without custom coupling code.

## Configuration

| Option                   | Description                                              | Default        |
|--------------------------|----------------------------------------------------------|----------------|
| `wall_eddy_diffusivity` | Wall eddy diffusivity controlling wall mixing [m^2/s].  | Required       |
| `chamber_radius`        | Radius of the spherical chamber [m] (spherical builder). | Required       |
| `chamber_dimensions`    | Length, width, height of rectangular chamber [m].        | Required (rectangular) |
| `distribution_type`     | `"discrete"`, `"continuous_pdf"`, or `"particle_resolved"`. | `"discrete"` |

## Best Practices

1. **Match distribution type to your builder**: Ensure `distribution_type` matches how your native `ParticleData` was constructed to avoid unintended behavior.
2. **Use builder/factory validation**: Set parameters through the builders or factory to enforce positivity, 3D chamber dimensions, and allowed distribution types with automatic unit conversion.
3. **Use physically reasonable parameters**: Choose `wall_eddy_diffusivity`, `chamber_radius`, or `chamber_dimensions` consistent with your experimental setup.
4. **Compose processes explicitly**: When combining wall loss with other dynamics, keep a clear, ordered time loop so you can reason about which processes act first in each step.

## Limitations

- Supports spherical and rectangular chambers; other geometries require new strategies/builders to be added to the factory.
- Factory selection is limited to registered strategy names.
- Does not include a high-level chamber orchestrator; you are responsible for building the main time-stepping loop that combines multiple strategies.

## Related Documentation

- **Hands-on guide**: [Wall loss tutorial](../../examples/processes/wall-loss/wall_loss_tutorial.ipynb)
- **Notebooks**: [Spherical wall loss strategy](../../examples/processes/wall-loss/spherical_wall_loss_strategy.ipynb)
- **Dynamics overview**:
  [Wall loss strategies](catalog.md#scientific-processes)

## FAQ

### Should I use the function-based or strategy-based wall loss API?

Use `SphericalWallLossStrategy` whenever you are already working with native `ParticleData` and other dynamics strategies. The function-based API remains available for lower-level or legacy workflows.

### How do I add a new wall loss model or geometry?

Subclass `WallLossStrategy` in your own code or contribute a new strategy to particula that implements `loss_coefficient` using the appropriate wall loss physics, then expose it through `particula.dynamics`.

## See Also

- [Condensation strategies](../../examples/simulations/index.md)
- [Particle phase examples](../../examples/fundamentals/particles/index.md)
