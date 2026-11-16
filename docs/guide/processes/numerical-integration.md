# Numerical integration and process splitting

Use adaptive integration when you need to control error in continuous
condensation rates. Use process splitting when you need an explicit order for
several deterministic processes. These are separate choices: an accurate
condensation step does not remove the error caused by alternating condensation
and dilution.

This guide describes development APIs imported from their concrete modules.
Existing `RunnableSequence.execute()` calls and prepared GPU simulations retain
their existing fixed-step behavior. No method is selected from the particle
state automatically. Start with [native process setup](index.md) and the
[condensation configuration guide](condensation.md).

## Choose a supported method

| Operation | CPU | Warp CPU / CUDA | Important boundary |
| --- | --- | --- | --- |
| Continuous isothermal condensation | `CondensationODE`, RK23 or ROS2 | `GPUCondensationODE`, RK23 | Adaptive mass rates, separate gas/particle tolerances |
| Existing finite-step condensation | Existing strategies | Existing direct/prepared fixed-four path | Inventory limits and fixed substeps; no embedded error estimate |
| Exact dilution | Existing strategy and `dilution_flow` | Existing direct/prepared path | Keep the exact exponential update |
| Grouped sequential or Strang composition | `SplitSimulation` | No resident adaptive split owner | Explicit deterministic component declarations |
| Equilibrium projections or stochastic events | Existing process-specific execution | Existing qualified prepared processes | Not accepted as adaptive ODE or Strang components |

For the measured growth, equilibrium and depletion cases, RK23 is the initial
choice. ROS2 remains an explicit CPU option for stiff continuous problems with
a suitable Jacobian; its lower-order error estimator can make it much more
expensive. The [method assessment](../../contribute/validation/numerics/adaptive-methods.md)
reports both accuracy and cost, including settings that failed the observable
accuracy target. A local tolerance is not a guarantee on global radius or vapor
error. Always refine tolerances and the outer split step for your own case.

## Integrate continuous condensation

Given a native `aerosol` and explicitly configured isothermal `strategy`:

```python
from particula.execution.condensation_ode import CondensationODE

simulation = CondensationODE(
    aerosol,
    strategy,
    method="rk23",
    rtol=1e-5,
    particle_atol=1e-27,  # kg per particle, for each material
    gas_atol=1e-15,       # kg/m3, for each gas species
    first_step=1e-3,      # s
    min_step=1e-14,       # s
    max_step=1.0,         # s
    max_trials=100,
)
while simulation.time < 10.0:
    simulation.step(10.0)
result = simulation.snapshot()
```

`step(bound)` accepts at most one step and never passes the absolute time
`bound`, in seconds. `snapshot()` returns detached accepted native state.
Construction detaches the input state and strategy. Mutating the input or a
snapshot does not modify the simulation.

`particle_atol` and `gas_atol` accept positive finite scalars or their full
native array shapes. Choose absolute tolerances from physically meaningful
small inventories; an excessively large absolute tolerance can hide the very
species you want to study. The controller uses the maximum across participating
components and boxes of

\[
\frac{|e_i|}{a_i+r_i\max(|y_i^{old}|,|y_i^{trial}|)}.
\]

All boxes share one accepted clock. The maximum norm prevents averaging a
small-species or fast-box error into a much larger reservoir. Internal scaling
improves conditioning but does not change the physical units of the tolerances.
Resolved populations use counts divided by representative volume for the gas
budget; discrete populations use number density. The CPU adapter also retains
native PDF quadrature weights. Inert gas lanes, inactive slots and unmapped
particle materials are preserved.

The CPU adapter accepts exact `CondensationIsothermal` and
`CondensationIsothermalStaggered` types. For the latter it uses the continuous
mass rate, not randomized theta choices or staggered finite-step sweeps.
Latent-heat strategies are excluded from this adapter. Full evaporation of an
active slot needs an explicit event/retirement policy and is not supported.
The existing minimum-radius physical-rate safeguard is not evidence for
sub-continuum validity.

Invalid inventory stages cause rejection and a shorter trial, without clipping.
Exhausting `max_trials` or reaching `min_step` raises an error. Accepted state
and time remain at the last successful step; previously accepted work is not
rolled back. `solver.statistics` records accepted/rejected trials, domain
rejections and numerical work. It counts Jacobian callbacks separately from
rate callbacks; a numerical Jacobian can itself evaluate many physical rates.

CPU ROS2 accepts dense or SciPy sparse Jacobians. Native condensation uses a
sparse Jacobian above 64 coordinates for the supported ideal mass/molar/volume
and kappa activity models with mass/molar/volume surface models. Particle-local
composition derivatives are grouped by material; gas rows retain the weighted
reservoir coupling. Other activity or surface implementations retain individual
column differences, since arbitrary callbacks may couple different particles.
Particle perturbations follow the current total particle mass, including dry
cores, so an initially wet scale does not distort derivatives after evaporation.
This reduces derivative work and storage for large populations; it does not
change the ROS2 error estimator or remove its accuracy cost.

The [wet–dry relaxation example](../../examples/simulations/adaptive_cloud_relaxation.py)
compares stiff evaporation with independent Radau/BDF references. The
[finite-reservoir example](../../examples/simulations/adaptive_reservoir_depletion.py)
and its [CUDA companion](../../examples/simulations/adaptive_reservoir_depletion_cuda.py)
compare radius and vapor errors during depletion. Both distinguish mass closure
from trajectory accuracy; the small CUDA case is not an acceleration benchmark.

## Explicit GPU execution and transfer costs

```python
from particula.execution.gpu_condensation_ode import GPUCondensationODE

simulation = GPUCondensationODE(
    aerosol,
    strategy,
    device="cuda:0",
    rtol=1e-5,
    particle_atol=1e-27,
    gas_atol=1e-15,
)
while simulation.time < 10.0:
    simulation.step(10.0)
result = simulation.snapshot()  # explicit full accepted-state download
```

The GPU profile accepts native discrete or resolved state, exact isothermal
strategy type, coupled gas updates and the existing mapped constant/Buck vapor
pressure and activity/surface profile. It permits one volatile material,
including a zero constant saturation pressure for involatile uptake. It does
not accept arbitrary CPU activity callbacks, latent heat, PDF, or ROS2. An
unavailable requested CUDA device fails; there is no CPU fallback.

Construction uploads state and configuration. Trial states, derivatives and
accepted arrays remain on the requested device. A successful trial normally
reads nine control scalars: stage-domain checks and the final error reduction.
Rejected stages can read fewer. Optional forcing uploads temperature and
pressure at each stage. The shared host controller synchronizes these reads;
this is not a fully asynchronous or CUDA-capturable adaptive loop.
`control_readbacks` exposes the count, `statistics` exposes numerical work, and
`device_storage_bytes` counts retained arrays, not measured peak GPU memory.

Calling `step()` or `snapshot()` during CUDA capture fails before launching
adaptive work. Device/configuration failures fault the owner; it must not be
reused. Physical trial-domain rejection instead permits retry. Use existing
qualified fixed-step prepared execution when capture/replay is required.

## Forcing and observation times

A pure `forcing(time)` callback may return `(temperature, pressure)`, in kelvin
and pascals, each a scalar or an array with one value per box. It is evaluated
at absolute integration stage times. It must not consume RNG, modify external
state, or depend on the order of rejected evaluations. Snapshot evaluation
also calls forcing at the accepted time.

For reproducible step or linear environmental histories, use the immutable
`EnvironmentForcing` table described in the
[recording and restart guide](../getting-started/recording-restart.md). Adaptive
owners and built-in condensation split flows land at table knots. Endpoint
stages use the previous segment; snapshots at a knot use its right-continuous
environment. Tables have a supported disk identity. Arbitrary discontinuous
callbacks still require caller-defined boundaries and one-sided values, and
cannot be saved as exact restart forcing.

Observe only after an accepted complete step or split cycle. Shortening a step
to land on an output time can change the adaptive trajectory. Compare runs
with different output schedules within the scientific accuracy target, rather
than expecting identical floating-point trajectories. Passive snapshots alone
do not change the accepted state or controller.

## Compose deterministic processes

```python
from particula.execution.splitting import (
    SplitGroup,
    SplitSimulation,
    condensation_flow,
    dilution_flow,
)

group = SplitGroup(
    (
        condensation_flow(
            "growth", strategy, rtol=1e-7,
            particle_atol=1e-27, gas_atol=1e-15,
        ),
        dilution_flow("ventilation", 1e-3),  # hazard in s^-1
    ),
    method="strang",
)
simulation = SplitSimulation(aerosol, group, time_step=0.1)
while simulation.time < 10.0:
    simulation.step(10.0)
result = simulation.snapshot()
```

Sequential composition advances each child for the full interval in the
listed order. Strang advances all but the last child for a half interval,
advances the last for the full interval, then repeats the outer children in
reverse order for the second half. Each child's absolute clock spans its own
physical interval. Nested groups are preserved; names must be unique across
the whole tree.

Strang requires declared component order of at least two, or an exact map.
It provides second-order composition only under the appropriate smoothness and
component-accuracy assumptions. Third-order component integration does not make
a generic Strang composition third order. A first-order nested sequential group
cannot be presented as a second-order component.

A failure discards the current complete cycle and leaves earlier cycles
accepted. Component controllers restart at each split boundary; both component
tolerances and `time_step` therefore need refinement. Recording is outside the
scheduler, so it never sees half cycles or rejected component trials.

Custom deterministic `Flow(name, advance, kind, order)` callbacks receive
detached native state, absolute start time and duration. They must return valid
native state and must not mutate external resources or consume RNG. Declaring
`kind="exact"` is a scientific assertion, not something the scheduler verifies.
Existing runnable `rate()` methods are not automatically continuous derivatives.
Stochastic updates and equilibrium projections need their own supported
scheduling contracts and are rejected as flow kinds here.
