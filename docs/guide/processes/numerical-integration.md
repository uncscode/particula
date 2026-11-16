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
| Continuous isothermal condensation | `CondensationODE`, RK23, ROS2 or ROS3 | `GPUCondensationODE` or `GPUResidentCondensation`, RK23 or ROS3 | Multiple exchanging materials; separate gas/particle tolerances |
| Existing finite-step condensation | Existing strategies | Existing direct/prepared fixed-four path | Inventory limits and fixed substeps; no embedded error estimate |
| Exact dilution | Existing strategy and `dilution_flow` | Existing direct/prepared path | Keep the exact exponential update |
| Grouped sequential or Strang composition | `SplitSimulation` | Fixed native events with adaptive condensation through `GPUCoupledSimulation` | Sequential composition; `resident=True` keeps stage handoff and adaptation on CUDA; no adaptive Strang owner |
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
strategy type and coupled gas updates. All mapped exchanging materials are
retained, with constant/Buck vapor pressures, ideal mass/molar/volume or kappa
activity, and static or phased mass/molar/volume surface tension. Surface
material densities are distinct from particle material densities. Temperature
surface tables, arbitrary CPU callbacks, latent heat, PDF and GPU ROS2 remain
unsupported. Requesting unavailable CUDA fails without a CPU fallback.

RK23 operates on every material and gas coordinate. ROS3 retains dense local
cross-material Jacobian blocks and the shared gas coupling. Colored finite
differences evaluate native device rates; a conservative block-arrowhead solve
uses storage linear in particle count and quadratic in exchanging materials.
It does not form a population-sized dense matrix. The single-exchange path
retains its scalar elimination. Unmapped dry materials and inert gases remain
unchanged. CPU RK23/ROS2/ROS3 also support multiple exchanging materials.

`GPUCondensationODE` keeps arrays and rates on device but uses a host controller.
A successful single-exchange trial normally reads nine control scalars for RK23
or nineteen for autonomous ROS3. Multiple-exchange ROS3 performs additional
finite-difference domain checks. Optional host forcing uploads temperature and
pressure at each rate evaluation. `control_readbacks` and `statistics` expose
this work; `device_storage_bytes` counts allocated arrays rather than measured
peak memory. This owner is not CUDA-capturable.

For device-owned adaptation, import `GPUResidentCondensation` from
`particula.execution.gpu_resident_condensation`. It accepts the same numerical
controls and either `method="rk23"` or `method="ros3"`. Native CUDA conditional
graphs require Warp's CUDA Toolkit and the driver to support CUDA 12.4 or later.
`advance(bound)` (also exposed as `step(bound)`) runs all internal trials to the
specified boundary. The accepted clock, error decision, rejection count and
next step stay on the GPU. Two small control records are downloaded after the
graph finishes; `snapshot()` explicitly downloads accepted physical state.
Capture during construction does not advance the simulation. There is no Warp
CPU or host-controller fallback. Arbitrary host forcing callbacks are rejected;
environment is constant inside each segment. Standalone resident adaptive
checkpoint serialization is not exposed; coupled boundary restart below is
supported.

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
    order="events-first", resident=True,
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

Each cycle retains the event owner's buffers and random streams. With
`resident=True`, a persistent `GPUResidentCondensation` owner imports event
state using device-to-device copies, resets scaled coordinates and its
controller on the GPU, integrates on CUDA and commits accepted masses/gas to
the same event buffers. `coupling_downloads` and `coupling_uploads` remain zero.
Host code schedules fixed events, forcing boundaries and observations, but
makes no adaptive acceptance or step-size decisions. Rejected trials never
repeat events or consume event random numbers. The deliberate controller reset
at each event boundary is part of exact continuation.

The default `resident=False` preserves the explicit-transfer path: two full
downloads and two uploads per cycle, with a host adaptive controller. Both
profiles aggregate `statistics` and `control_readbacks`, and provide explicit
observations and checkpoints. The checkpoint records the selected profile and
rejects old numerical schemas. `capture_events()` captures the fixed event
sequence; the resident adaptive owner separately captures its conditional loop.
The whole Python cycle, observations and checkpoint serialization remain outside
CUDA capture.

The [cyclic chamber](../../examples/gpu/prepared/gpu_prepared_cyclic_chamber.py)
selects resident execution. The older
[forcing study](../../examples/gpu/prepared/gpu_prepared_forcing.py) retains its
explicit-transfer policy. Their scientific schedules and stochastic errors are
separate from condensation's local error control.

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

### Selective forcing of a coupled GPU simulation

`GPUCoupledSimulation.update_gas_species()` accepts dictionaries keyed by gas
name. `concentrations` supplies kg/m³; `saturation_ratios` supplies dimensionless
ratios. Each value is a scalar or per-box vector. A species may have only one
target. Unspecified gas species remain untouched on the device. Saturation
forcing evaluates the configured constant/Buck law at the device's current
temperature; it does not download gas or temperature to compute a new target.
These are explicit external forcing boundaries, separate from adaptive trials.

```python
simulation.update_gas_species(
    {"precursor": 1e-9}, saturation_ratios={"water": 1.02}
)
```

All targets validate before physical mutation. A scalar device status check
rejects nonphysical derived concentrations. Requested observations can still
check an independent external mass ledger on the CPU.
