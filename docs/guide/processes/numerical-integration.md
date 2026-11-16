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
| Continuous isothermal condensation | `CondensationODE`, RK23, ROS2 or ROS3 | `GPUCondensationODE`, RK23 or ROS3 | Adaptive mass rates, separate gas/particle tolerances |
| Existing finite-step condensation | Existing strategies | Existing direct/prepared fixed-four path | Inventory limits and fixed substeps; no embedded error estimate |
| Exact dilution | Existing strategy and `dilution_flow` | Existing direct/prepared path | Keep the exact exponential update |
| Grouped sequential or Strang composition | `SplitSimulation` | Fixed native events with adaptive condensation through `GPUCoupledSimulation` | GPU composition is sequential with explicit host transfers; no resident adaptive Strang owner |
| Equilibrium projections or stochastic events | Existing process-specific execution | Existing qualified prepared processes | Not accepted as adaptive ODE or Strang components |

For the measured growth, equilibrium and depletion cases, RK23 is the initial
choice. ROS2 and ROS3 are explicit CPU options for stiff continuous problems
with a suitable Jacobian. ROS3 uses a third-order solution and second-order
embedded estimator; it substantially reduces accepted steps in the measured
wet–dry relaxation case at the same tolerances. ROS2's lower-order estimator
can be expensive. The [method assessment](../../contribute/validation/numerics/adaptive-methods.md)
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
For ROS2 and ROS3, each stage solve includes two residual corrections using
the same factorization. The linear-solve counter includes those corrections.
They reduce roundoff contamination of passive inventories; they do not clip
small physical rates or replace the error estimator. Exactly unchanged scaled
coordinates retain their initial physical values when converted for observation.

CPU ROS2 and ROS3 accept dense or SciPy sparse Jacobians. Native condensation uses a
sparse Jacobian above 64 coordinates for the supported ideal mass/molar/volume
and kappa activity models with mass/molar/volume surface models. Particle-local
composition derivatives are grouped by material; gas rows retain the weighted
reservoir coupling. Other activity or surface implementations retain individual
column differences, since arbitrary callbacks may couple different particles.
Particle perturbations follow the current total particle mass, including dry
cores, so an initially wet scale does not distort derivatives after evaporation.
This reduces derivative work and storage for large populations; it does not
change the chosen method's error estimator or remove its accuracy cost.

The [wet–dry relaxation example](../../examples/simulations/adaptive_cloud_relaxation.py)
compares stiff evaporation with independent Radau/BDF references. The
[CUDA wet–dry example](../../examples/simulations/adaptive_cloud_relaxation_cuda.py)
qualifies device ROS3 against the same references and explicit humidity reset. The
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
not accept arbitrary CPU activity callbacks, latent heat, PDF or ROS2. An
unavailable requested CUDA device fails; there is no CPU fallback.

Select `method="ros3"` for the qualified stiff profile with exactly one mapped
exchanging material. Its local particle derivatives and one gas reservoir per
box form an arrowhead Jacobian: particle coordinates couple only through their
shared gas. Colored finite differences use the native device mass rates;
conservative elimination solves each stage in linear storage without a dense
matrix or CPU rate evaluation. Other mapped exchanges are rejected before
device allocation. Unmapped dry material and inert gases remain unchanged.
The method uses float64 coefficients, a second-order embedded estimate, and
the same maximum error norm and rejection policy as CPU ROS3. This does not
make an enclosing fixed stochastic schedule adaptive or CUDA-capturable.

Construction uploads state and configuration. Trial states, derivatives and
accepted arrays remain on the requested device. A successful trial normally
reads nine control scalars for RK23 or nineteen for autonomous ROS3:
domain/linear-solve checks and the final error reduction.
Rejected stages can read fewer. Optional forcing uploads temperature and
pressure at each rate evaluation. ROS3 also differentiates forcing at fixed
state inside the current smooth interval, adding one device rate evaluation
and two scalar checks. The shared host controller synchronizes these reads;
this is not a fully asynchronous or CUDA-capturable adaptive loop.
`control_readbacks` exposes the count, `statistics` exposes numerical work, and
`device_storage_bytes` counts retained arrays, not measured peak GPU memory.
ROS3 reports every physical rate call used by its finite differences, one
Jacobian assembly per trial and three linear solves. It performs no LU
factorization; its factorization counter is therefore zero.

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

## Fixed GPU events with adaptive condensation

`GPUCoupledSimulation` from `particula.execution.gpu_coupled` composes a fixed
native event sequence with a supported `GPUCondensationODE` flow. It is useful
when stiff condensation needs smaller error-controlled steps while sources,
collisions or removal retain their chosen physical event schedule. For an
existing native `aerosol`, configured `source`/`coagulation` runnables and an
isothermal condensation `strategy`:

```python
from particula.execution.gpu_coupled import (
    GPUCondensationControls,
    GPUCoupledSimulation,
)

with GPUCoupledSimulation(
    aerosol, (source, coagulation), strategy,
    device="cuda:0", time_step=1.0, root_seed=42,
    order="events-first",
    controls=GPUCondensationControls(method="ros3", rtol=1e-5),
) as simulation:
    simulation.capture_events()
    simulation.run(10)
    result = simulation.snapshot()
    checkpoint = simulation.checkpoint()
```

Use `order="condensation-first"` when condensation precedes the fixed events,
for example exact dilution. This is sequential composition; condensation's
local error estimate does not measure splitting or stochastic uncertainty.
The complete event interval remains fixed. A shorter requested bound rejects
before any physical work, so forcing and output schedules must use that grid.
Temperature and gas replacements are explicit calls between complete cycles.

Each cycle retains the event owner's buffers and random streams. It downloads
native state, uploads a detached adaptive problem, downloads accepted masses
and gas, then copies those fields into the existing event buffers. Physical
rates run on the requested device. `coupling_downloads` and `coupling_uploads`
count these full transfer boundaries (normally two each per cycle), excluding
user-requested observations and checkpoints. `statistics` and
`control_readbacks` aggregate condensation work. Rejected trials never repeat
an event or draw event random numbers. Adaptive buffers and the controller are
rebuilt at each event boundary; this policy is part of exact continuation.

`capture_events()` captures only the fixed event sequence. Calling the whole
cycle, observation or checkpoint inside an active CUDA capture rejects. The
host-controlled composition makes no fully resident or acceleration claim.
The [cyclic chamber](../../examples/gpu/prepared/gpu_prepared_cyclic_chamber.py)
and [forcing study](../../examples/gpu/prepared/gpu_prepared_forcing.py)
retain their full scientific schedules with this explicit transfer policy.

Supported events are the existing prepared nucleation, coagulation, wall-loss
and dilution families, at most one of each. Their normal species, population,
capacity and equal-count restrictions still apply. Saturation-dependent
nucleation and extension fields are currently rejected. Observed saturation
is derived from the configured vapor-pressure law; it is not an input to the
fixed event sequence. Empty populations preserve gas/event continuation and
skip condensation until a source provides active particles.

The owner publishes time and observations only after a complete cycle. If
condensation or an upload fails after events have executed, the owner faults
and rejects subsequent steps, snapshots and checkpoints. Earlier event writes
are not rolled back. Resume the last successful checkpoint explicitly.

`resume_coupled`, `save_coupled_checkpoint` and `load_coupled_checkpoint` from
`particula.execution.gpu_coupled_restart` retain canonical event/RNG bytes,
condensation configuration, controls, order and work counters. Supply the same
event tuple and strategy and an explicitly matching device to `resume_coupled`.
The versioned archive contains data only; analysis snapshots cannot replace
it. Future forcing/source schedules remain the caller's responsibility.
