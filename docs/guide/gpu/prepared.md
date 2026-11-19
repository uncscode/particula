# Prepared GPU simulations

Build a native CPU `Aerosol`, choose supported dynamics, and call `prepare()`
to upload its particle, gas and environment state once. The returned
`PreparedSimulation` owns a fixed GPU workspace. Repeated steps stay on that
device; `snapshot()` returns a detached CPU aerosol for analysis and plotting.
The initial aerosol remains unchanged.

The [complete quick start](../../examples/gpu/prepared/gpu_prepared_quick_start.py)
constructs all inputs and plots the resulting distributions. With an existing
native `aerosol` and configured `condensation` and `dilution` runnables:

```python
from particula import Backend

processes = condensation | dilution
with processes.prepare(
    aerosol, backend=Backend.WARP, device="cuda:0",
    time_step=1e-6, root_seed=42,
) as simulation:
    simulation.run(steps=1000)
    simulation.update_environment(temperature=283.15)
    simulation.capture()
    simulation.run(steps=2000)
    summary = simulation.diagnostics()
    checkpoint = simulation.checkpoint()
    result = simulation.snapshot()

with processes.resume(checkpoint) as continuation:
    continuation.run(steps=1000)
    later = continuation.snapshot()
```

This is explicit CUDA execution. Warp must be installed and the selected GPU
available; an unavailable target raises instead of substituting CPU execution.
`Backend.WARP` with `device="cpu"` is a separate regression path. Native NumPy
CPU execution continues to use the existing runnable execution methods.

## Supported process plans

Preparation validates exact supported classes and their configuration before
upload. Compose ordered subsets with `|`, with one occurrence of each family.
The order is scientifically meaningful. Supported preparation is narrower than
the full collection of CPU strategies or direct GPU mechanism masks.

| Family | Prepared support | Boundaries |
| --- | --- | --- |
| Condensation | `MassCondensation` with isothermal or latent-heat strategy | Native discrete or resolved populations; explicit species maps; supported ideal, κ and homogeneous binary BAT activities and constant/Water Buck vapor-pressure models; four fixed equal GPU substeps. |
| Coagulation | `Coagulation` with Brownian, charged, SP2016 sedimentation, ST1956 shear, or their combined strategy | Particle-resolved, equal positive active counts, default interpolation controls; at most one of each mechanism. A charged strategy selects its fit; combining it explicitly with Brownian sums both terms. |
| Dilution | `Dilution` with exact exponential strategy | Counts/densities and gas inventory decrease consistently; no volume change. |
| Wall loss | `WallLoss` with spherical, rectangular or charged strategy | Particle-resolved; supported neutral/charged configuration is frozen at preparation; retired resolved slots retain the native raw-count semantics. |
| Nucleation | `Nucleation` with activation or kinetic strategy | Resolved unit-count, neutral injection with explicit species mapping, integral event demand and fixed capacity; supported exhaustion/resampling policy is retained. |
| Ion source | `IonPairSource` with constant production | Paired Poisson unit-count injection into fixed capacity; independent source RNG and inventory ledger. |
| Ion cleanup | `IonNeutralCleanup` | Exports neutral ion-only products from reserved material lanes; retains neutralized aerosol particles and records exported mass. |
| Ion condensation exclusion | `IonCondensationExclusion` | Companion configuration gating ion-only vapor proposals before inventory limiting in every substep. |
| Seed cohort | `SeedCohort` | Original unit-count seed lineages; first passage before background capture; observes after every physical process. |

Prepared charged coagulation accepts `HardSphereKernelStrategy`,
`CoulombDyachkov2007KernelStrategy`, `CoulombGatti2008KernelStrategy`,
`CoulombGopalakrishnan2012KernelStrategy`, and `CoulumbChahl2019KernelStrategy`
inside `ChargedCoagulationStrategy`. The selected fit is frozen and included in
checkpoint compatibility. Each uses the shared stochastic sampler and bounded
parallel pair maximum reduction. Capture and resume use the same model.
Custom subclasses and unsupported interpolation controls reject before upload.

Pass `coagulation_work_config=CoagulationWorkConfig(...)` to `prepare()` to
set the per-box trial budget and collision-buffer byte limit. Import this
configuration from `particula.gpu.kernels.coagulation`. Budgets and sampler
version are frozen in checkpoints. All 15 nonempty mechanism subsets share
exact tiled pair reduction with quadratic work and bounded scratch; population
size has no algorithmic cap. Slot indices remain signed int32. Prepared
sedimentation retains the CPU strategy's mean material density convention;
the direct kernel uses each particle's effective density.

By default, each step admits conservative trial demand before touching coagulation outputs,
particles or RNG. Excessive demand latches a failure; `synchronize()`,
`snapshot()` or another observation boundary reports it. Reduce the configured
timestep and prepare a fresh owner. Adaptive admission is available through the optional policy below. See
[direct configuration and preflight](direct/coagulation-configuration.md).
See the [charged model comparison](../../examples/gpu/prepared/gpu_charged_models.py).
The [charged coagulation validation record](../../contribute/validation/gpu/charged-coagulation.md)
lists actual CUDA checks and remaining scientific boundaries.
For explicit atmospheric ions, use the separate
[atmospheric collision configuration](#atmospheric-ion-collisions) below.

Condensation retains configured transport coefficients and uses the mapped vapor
material's molar mass for transport mean free paths, matching native CPU physics.
It requires compatible
material/surface density, surface-tension model and molar masses. Ideal mass or
volume activity has narrower single-material/inactive support. Unsupported
custom classes, PDF populations, staggered/adaptive stepping, multiple volatile
material mappings or unsupported thermodynamics reject explicitly. See the
[direct condensation configuration](direct/condensation-configuration.md) for
underlying physics and the executable preparation error for a rejected plan.
Latent energy is diagnostic bookkeeping, not automatic temperature feedback.
For the supported single-phase volume-weighted surface model, each particle
uses its own composition and requires uniform physical material densities;
separate surface-weighting density translation is not implemented. CPU surface
mixing and Kelvin terms preserve that particle axis as well; independent rows
are never averaged into one mixture.

Process configuration, timestep, box layout and buffer capacity are frozen.
Editing the original Python strategy after preparation does not change a running
simulation. Changing these requires a new preparation; a checkpoint resume also
requires matching configuration and the exact supported device.

## Adaptive coagulation intervals

Use an adaptive interval when a physical box outgrows its original trial
budget during a run. Each admitted pass recomputes the rate majorant from the
current particles. All boxes first try the time remaining to the observation.
The controller halves this duration until every box can schedule its full
demand. It consumes random draws only after admission. The final duration
covers the observation interval within eight fp64 ULPs. Timestep convergence
and sampling of rare particles require separate scientific checks.

```python
from particula.gpu.kernels.coagulation import (
    CoagulationSubcyclingConfig, CoagulationSubcyclingError,
    CoagulationWorkConfig,
)

policy = CoagulationSubcyclingConfig(
    min_substep=0.001, max_subcycles=32, max_device_bytes=64 * 1024**2
)
with processes.prepare(
    aerosol, device="cuda:0", time_step=1.0,
    coagulation_work_config=CoagulationWorkConfig(max_trials_per_box=256),
    coagulation_subcycling=policy,
) as sim:
    try:
        sim.step()
    except CoagulationSubcyclingError as error:
        print(error.diagnostics["reason"], error.diagnostics["remaining_seconds"])
        committed = sim.checkpoint()
        with processes.resume(
            committed, retry_subcycling=CoagulationSubcyclingConfig(
                min_substep=0.0001, max_subcycles=128, max_device_bytes=128 * 1024**2
            ),
        ) as retry:
            retry.step()
            print(retry.coagulation_diagnostics()["substep_sizes"])
```

The trial cap, allocated collision slots and half the active slot count all
limit a pass. The byte cap on the scratch pair buffer determines its capacity
at setup. The device-memory ceiling covers a conservative reservation for
Particula's live arrays, rollback copies, bounded history and ordered pair
log. A one-pair-per-box buffer is the minimum. The reservation can exceed the
eventual allocation when a process shares scratch arrays. CUDA context,
allocator pool, compiled modules and graph runtime storage are additional
driver costs. No arrays grow during a step.

A prepared step saves device state before running the first process and reads
one compact commit status at the end. Condensation, sources, dilution and
wall loss each run once at their declared positions with the outer timestep.
Field and cohort updates follow every admitted merge. A subcycling limit
restores the entire attempted interval on device, including earlier processes,
pair buffers and RNG. Earlier steps in `run(n)` stay committed. Custom process
adapters are excluded because their external writes cannot be restored.
Physical-domain and unexpected execution errors follow the terminal fault path.

`coagulation_diagnostics()` returns a row per accepted pass and box, including
majorant, active slots, remaining-interval demand, accepted-duration demand,
work limits and their ties, scheduled and attempted trials, and actual mergers.
`include_pairs=True` downloads the ordered interval log. For pass i in box b,
slice from the previous `pair_end_offsets[i-1,b]` (zero for the first pass)
to `pair_end_offsets[i,b]`. The device scratch pair buffer holds the last pass.
`last_rejection` preserves tentative counts separately, marked `rolled_back`. Its `last_*` fields
include the last candidate duration, demand for the remaining interval,
active counts and controlling capacities, even on a first-pass rejection.
`allocation_bytes` reports retained live and restore storage. Fresh owners
collect diagnostics starting at their first step.

Adaptive checkpoints freeze the controller version, policy, outer timestep and
sampler configuration. Ordinary resume uses the same random streams and
policy. An explicit retry can lower the minimum, raise the pass or memory
ceiling, or combine these changes. The new owner records the change in
`policy_change` and requires explicit capture for graph execution. Captured
steps use the same bounded device decisions as ordinary steps. Once a step
finishes, unused passes skip physics and RNG. A recoverable rejection keeps
the existing owner's graph valid. The worked example follows two cloud boxes
through exhaustion and retry. [Adaptive cloud coagulation](../../examples/gpu/prepared/adaptive_coagulation.py)

## Atmospheric ion collisions

`AtmosphericCoagulation` from `particula.dynamics.ions.coagulation` selects
one collision law per pair within the existing prepared sampler:

- aerosol–aerosol: the selected `charged_model`;
- free ion–aerosol: the conductive-air LF2013 attachment coefficients;
- opposite free ions: a constant recombination coefficient, default
  `1.7e-12 m³/s`;
- same-sign free ions and neutral ion-only products: zero collision rate.

This is one coagulation process, so do not compose a second `Coagulation` with
it. Supply distinct `positive_species` and `negative_species` material lanes.
Active particles must have unit represented counts. Positive ions have mass
150 u and charge +1; negative ions have mass 90 u and charge −1. The module
exports `POSITIVE_ION_MASS` and `NEGATIVE_ION_MASS` in kg for source setup.
Choose physical material densities to obtain the intended free-ion diameter;
these densities continue to determine volume after attachment.

LF2013 is restricted to 298.15 K and 101325 Pa, integer aerosol charge between
−100 and +100, and the published radius interval for each charge and polarity.
The global radius domain is 0.2 nm–10 µm. State and needed attachment rates are
validated on the device before collision sampling; invalid states or unsupported
fits fault at observation without advancing the collision RNG or applying merges.
A prior process in the same ordered step may already have completed.

The default `omitted="reject"` rejects repulsive rates below the table's fitted
radius interval. `omitted="lower"` assigns zero and `omitted="upper"` assigns
the published omission cutoff, `1e-15 m³/s`, only in those omitted repulsive
regions. These are explicit sensitivity bounds, not an extrapolation. Report
both when using this uncertainty treatment. Other unsupported inputs still
reject. Fits are evaluated only for polarities currently present in a box.

Ion source, cleanup and condensation-exclusion material maps must agree.
Sources must use the reference ion masses. Condensation requires the explicit
`IonCondensationExclusion` companion. Recombination products conserve their
material masses until cleanup exports them; neutralized aerosols remain active.
The collision law supports CUDA capture and checkpoint continuation. Derived
classifications and rates are rebuilt from physical state each step.

The [ion-balance example](../../examples/gpu/prepared/gpu_ion_balance.py)
checks production, recombination and export against an independent finite-box
reference. The [validation record](../../contribute/validation/gpu/charged-coagulation.md)
distinguishes pair-law, conservation and finite-population decay evidence from
the [Lucky Few sensitivity study](../../examples/gpu/prepared/gpu_lucky_few.py).
Its rate assumptions do not guarantee bounds on survival probability.
The coefficients follow
[López-Yglesias and Flagan (2013)](https://doi.org/10.1080/02786826.2013.783684).

## Paired ion production

`IonPairSource` from `particula.dynamics.ions.source` is an explicit,
prepared-only particle source. It composes with runnables using `|`. Supply
`production_rate` in ion pairs m^-3 s^-1, distinct positive/negative material
indices, and each ion's mass in kg. It samples a Poisson number of pairs per
box per step, with mean `production_rate * volume * time_step`. Each event
inserts both polarities with unit count and charges +1 and -1.

The source owns a separate random stream keyed by the root seed and logical
box identity. It participates in native CUDA capture and exact checkpoint
continuation. `simulation.ion_source_diagnostics()` returns detached cumulative
`injected_pairs` and extensive `injected_mass` [kg, box by particle species].
Include these external inputs in conservation checks.

Free slots must accommodate both members of every sampled pair. Insufficient
capacity faults the simulation at observation; no partial pair, resampling or
volume scaling occurs. The Poisson mean is bounded at one million pairs per
box per step and uses Warp's float32 Poisson parameter. Ordinary particle
state and inventories remain fp64. Zero production advances no source RNG.
CPU `execute()` is unsupported and rejects explicitly.

This source supplies particles, not atmospheric ion collision physics. Do not
interpret ordinary charged coagulation of these particles as validated
ion–aerosol attachment or ion–ion recombination.

Compose `IonCondensationExclusion` from `particula.dynamics.ions.condensation`
with `MassCondensation` to keep free ions and neutral ion-only products out of
vapor transfer. Specify the reserved positive/negative material indices; they
must match any ion source/cleanup configuration and must not be condensation
targets. Particles containing any other material remain eligible, including
aerosols carrying attached ions. Uptake and evaporation proposals are zeroed
for ion-only particles before gas inventory limiting in each substep. Physical
particle counts, masses and charges are never temporarily masked.

This configuration is frozen at preparation and preserved by capture/restart.
The native minimum-radius setting is a transport floor, not an exclusion rule.
Prepared condensation with a static surface-tension model supports different
physical material densities with a constant surface-model density for Kelvin
curvature. Composition-weighted surfaces reject mixed material densities before
upload because CPU surface weights and device physical-volume weights differ.
Physical particle volume still sums mass divided by each material's density.
Nonuniform surface-model
densities remain unsupported. This distinction allows equal-diameter ions of
different molecular masses without changing their physical material densities.

`IonNeutralCleanup` from `particula.dynamics.ions.cleanup` removes neutral
products containing both reserved ion materials and no other material.
Supply the same positive/negative material indices as the source. These lanes
must be reserved for ions; material composition is the classification contract.
Neutral aerosol particles with any mass in another lane are retained. Cleanup
does not itself model a recombination reaction or remove charged free ions.

Place cleanup at the chosen point in the ordered composition. By default it
exports every prepared step, reclaiming slots by zeroing all particle fields.
Set `interval_steps=N` for export on steps N, 2N, and so on; the physical interval
is `N * time_step`. Its phase survives capture and checkpoint continuation.
Unit-count validation still runs every step. Neutral products remain inert but
occupy slots between exports, so capacity must accommodate the entire interval.
Source-before-cleanup ordering can fault on capacity before an export later in
that same step.
`simulation.ion_cleanup_diagnostics()` observes detached cumulative
`exported_particles` and `exported_mass` [kg, box by species]. Exported charge
is exactly zero. Check each species' initial mass plus injected mass against
remaining mass plus exported mass. The ledgers participate in CUDA capture and
exact checkpoint continuation. Unit active counts are required throughout;
evolving fractional counts fault before a box is modified.

## Original seed lineages

`SeedCohort` from `particula.execution.cohort` adds an observer to a prepared
composition. Supply distinct `seed_species` and `background_species` indices
and `target_diameter` [m, default 50 nm]. At preparation, each active particle
containing seed material contributes one original lineage; particles containing
background material receive a persistent background label. Labels describe the
initial population, not a later test of material composition.

The observer runs after every physical process, regardless of where its
configuration appears in the composition. It follows successfully applied
coagulation pairs, adds unresolved lineage counts on seed–seed mergers, and
clears donor labels before slot reuse. Any background merger captures unresolved
lineages before the merged particle's diameter is tested. A lineage that already
reached the threshold remains successful after later mergers. Diameter uses the
sum of material volumes, consistent with the native spherical-particle model.

`simulation.cohort_diagnostics()` returns detached per-box arrays `initial`,
`reached`, `captured` and `unresolved`. The invariant is
`initial = reached + captured + unresolved`. At the chosen end time, unresolved
lineages are right-censored. These counts differ from surviving particle counts:
two seeds may become one particle while retaining two unresolved lineages.
The labels and outcome ledger survive native capture and exact checkpoint
continuation; they never modify physical mass, charge or the collision RNG.

Supported physical companions are condensation, coagulation, ion sources and
ion cleanup. Ion material lanes must be distinct from both cohort markers.
Removal, dilution and resampling processes reject because their lineage-outcome
semantics are not defined here. All active counts must remain one.
First passage is resolved at process boundaries, not interpolated within a
condensation substep or physical time interval. A survival study must test
timestep and process-order sensitivity before interpreting rare-event results.

## Registered custom processes

Register an exact process class in a caller-owned `ProcessRegistry` and pass it
as `process_registry=` to `prepare()` and `resume()`. Registration does not
implicitly apply to subclasses. An adapter implements two public methods:

- `plan(process, aerosol)` validates detached CPU state and returns a frozen
  `GPUProcessPlan` with instance name, extension ID/version, read/write names,
  immutable parameters, capabilities and optional persistent-state descriptors.
- `prepare(plan, context)` receives a `GPUPreparationContext` and returns a
  `PreparedOperation` retaining a launch-only handler and its resources.

The context exposes declared device arrays, the device and stream, timestep,
logical box IDs, a device clock and shared failure flag. It exposes no private
simulation owner. Allocate scratch during preparation. The retained operation
must not allocate, synchronize, download data or reseed streams during a step.
Callbacks are trusted numerical implementations; declarations are not a Python
or device-memory security boundary.

Built-in physics uses the same exact-type registration and retained operation
records. Its internal preparation context supplies native buffers, resource
allocation and configuration without exposing simulation lifecycle methods.
Composition preserves operation capabilities and resource identities through
particle-field accounting, cohort observation and thermodynamic refreshes.

Declare `GPUStateSpec` arrays for persistent counters or other continuation
state. Access them through `context.state`. Scratch buffers are reconstructed
on resume and are not checkpointed. Checkpoint schema v2 stores immutable
configuration and bytes, not Python implementations. Resume matches the
caller-supplied registry and validates the payload inventory before upload.
Built-in-only v1 checkpoints remain readable with a zero initial time.

For managed Warp random streams, set `GPUProcessPlan(stochastic=True, ...)`.
`context.rng` then supplies one uint32 state word per logical box. One kernel
thread per box loads that word, draws with Warp's random functions, and writes
the advanced word back. Streams depend on root seed, extension ID, instance
name and logical box ID; reordering boxes or processes does not reseed them.
Checkpoint/resume preserves these words exactly, and capture records without
consuming draws. A zero timestep skips custom advancement. NumPy host random
streams and Warp streams have distinct algorithms and continuation formats.

Kernels must check `context.failure[0]` before writing and suppress work if it
is nonzero. They can report their own failure using codes 100 or greater.
Particula adds device-side finite/bounds checks for declared destinations and
floating persistent state. These checks latch failure before later guarded
operations, including subsequent captured replays; observation reports the
failure and faults the owner. Earlier writes are not rolled back. Integer
field checks retain integer precision, including int64 values above 2**53.

`time=` selects the initial absolute time. `current_time` includes that origin;
`simulated_time` remains elapsed time. Custom kernels read the current interval's
start from `context.clock[0]`, so captured replay follows evolving time.

## Registered particle fields

Prepared execution uploads `aerosol.fields` once and includes detached field
values and inventory ledgers in snapshots and checkpoints. Particle fields
require a resolved population. Their schema and device identities remain fixed
until a new preparation.

Additive float64 fields follow actual native collision pairs and the exact
represented-count overlaps used by native resampling. Dilution retains the
per-particle value while recording removed inventory. Wall loss and condensation
retirement clear freed slots and record removed inventory; nucleation initializes
actual newborn slots from `FieldSpec.initial_value` and records source inventory.
For each field, compare `sum(count * value)` with the initial inventory plus
`fields.inventory_changes(name)["source"]` minus its `"removal"` ledger. Vector
fields retain a separate budget for each particle-species component.

Passive fields, including int32/int64 labels, support births and retirement but
reject coagulation and enabled resampling because their merge rule is undefined.
Particle fields reject representative-volume scaling and ion source/cleanup
sequences without a supported field-event contract. Custom processes may change
values but must preserve slot activity. Active slots obey the declared bounds;
inactive slots must contain zero even when the active minimum is positive.

Field lifecycle operations retain their buffers during preparation and support
resident CUDA capture and exact checkpoint continuation. These accounting rules
preserve represented inventory; resampling can still alter the joint
particle/tracer distribution. Custom scientific sources and sinks remain the
custom process author's responsibility.

## State, forcing and observations

`run(steps)` enqueues fixed-duration steps. `completed_steps` and
`simulated_time` track enqueued work; synchronize or observe to establish device
success. Resident prepared steps perform no simulation-state download or device
array allocation. Explicit `CPUTransfer` operations incur host round trips. Explicit observation and forcing boundaries wait for pending
work and check status.

| Operation | Result and cost |
| --- | --- |
| `synchronize()` | Waits for this owner's stream and checks device failure status. |
| `diagnostics()` | Downloads box/species reductions: number, charge and mass concentration, active slots, thermodynamics and optional last-step latent energy. Does not download every particle. |
| `snapshot()` | Downloads detached native CPU particle, gas and environment arrays, retaining gas names, physical volumes and population measure. |
| `update_environment(...)` | Replaces positive temperature [K] and pressure [Pa], accepting scalars or per-box vectors. Without condensation, saturation can be prescribed explicitly. |
| `update_gas(concentration)` | Replaces the full ordered gas vector or box-by-gas matrix [kg/m³], in `gas_names` order. |
| `checkpoint()` | Saves immutable continuation bytes, RNG streams, counters, auxiliary state and frozen process configuration. Also provides detached inspection state. |
| `close()` | Drains outstanding work and releases resources; does not download a final snapshot. Context managers call it automatically. |

With condensation, saturation is derived from current gas and temperature using
the configured vapor-pressure law. Prescribing it separately rejects. Forcing
updates change values in place and refresh derived thermodynamics. They do not
change the immutable plan. Gas replacement is an external source or sink: record
its actual inventory change when checking conservation. Volume remains owned by
`ParticleData`, not the environment.

Snapshots are independent analysis objects. Editing one cannot change the GPU,
the initial CPU aerosol or another snapshot. A snapshot alone does not preserve
random-stream continuation. Checkpoints do; their canonical bytes, rather than
editable inspection arrays, govern restart. Checkpoints are currently in-memory
records, not a promised portable on-disk format. External forcing schedules and
their current position remain caller-owned.

## Capture and random streams

`capture()` records one timestep without advancing physics. Later `step()` and
`run()` replay the graph; `uncapture()` waits and restores ordinary prepared
execution. Capture requires native CUDA. The graph belongs to one live owner,
with fixed array identities, shapes, process order and timestep. Forcing may
update values between segments. Resume creates fresh buffers and requires a new
explicit capture. Warm up the first replay when timing: Warp may instantiate
the executable then.

For stable random-stream identities, supply a root seed and unique string
`logical_box_ids`, for example `("chamber-a", "chamber-b")`. Logical box identity
and process namespace determine streams, independently of physical box order.
Checkpoints preserve stream words and counters. Exact continuation is qualified
on the same device/runtime; cross-device bitwise reproducibility is not promised.

## Evolving compatibility and failure

Brownian coagulation requires equal active represented counts at every step.
For example, dilution can reduce existing counts while a later unit-count source
creates unequal cohorts. Such a sequence can become incompatible even though its
initial state was valid. A device guard stops coagulation before consuming its
RNG, and observation reports the failure. The cyclic chamber example omits
weight-changing sinks so repeated unit-count injection remains compatible.

Device failures stop dependent writers and fault the owner when observed.
Previously completed writes are not rolled back. Use a prior checkpoint to
restart a failed experiment; do not treat a faulted owner as reusable state.
Nucleation capacity and integral-demand failures likewise remain explicit.

## Choosing a useful workload

Small jobs can be dominated by preparation, transfers and graph setup. Keep
several steps resident and request reduced diagnostics when full distributions
are unnecessary. The Brownian majorant uses bounded temporary storage, but pair
work remains quadratic in particles per box. Large aggregate populations spread
across many independent boxes can therefore be more practical than one very
large coagulating box. Condensation and dilution have different scaling.

The [GPU examples](../../examples/gpu/index.md) show heterogeneous boxes,
time-varying forcing, stochastic ensembles, repeated source pulses and a
million-particle lifecycle. Their saved CUDA figures are deliberately refreshed;
site builds do not run these experiments.

## Runtime basis

This implementation is qualified with Warp 1.11.1. Warp arrays live on a chosen
device, kernel inputs must share that device, and CUDA graphs replay recorded
operations. Those constraints motivate explicit upload, stable owned buffers
and a graph tied to one simulation. See the matching
[Warp runtime documentation](https://nvidia.github.io/warp/v1.11/user_guide/runtime.html).
Our API's process support and exact continuation guarantees are established by
Particula tests, rather than inferred from Warp's broader API.

GPU launches are asynchronous, so timing only Python dispatch misses device
work. The campaign waits for the measured work before stopping its timer and
reports separate preparation and transfer costs. Nsight Systems supplies
kernel-level attribution; profiling runs are separate from timing repetitions.
See [Warp profiling guidance](https://nvidia.github.io/warp/v1.11/deep_dive/profiling.html).

The [two-million-particle environmental sweep](../../examples/gpu/prepared/gpu_environment_sweep.py)
combines charged Brownian coagulation with closed water condensation across
sixteen temperature, pressure and initial-humidity conditions. It keeps all
2,097,152 initial particles resident together and documents its constant
vapor-diffusivity approximation.

### Variable-temperature/pressure ion attachment

`AtmosphericCoagulation(..., attachment_model="mobility")` selects an explicit
Coulomb transport approximation instead of the reference-only LF2013 tables.
It uses each box's current temperature, pressure, wet aerosol radius, total
mass and charge; rates refresh before collision sampling, including graph
replay and restart. `omitted` sensitivity choices apply only to LF2013.

Ion friction is `e / Z`, with
`Z = Z_ref * sqrt(T / 298.15) * 101325 / P`. This scaling assumes a constant
momentum-transfer collision cross section in the low-field Mason–Schamp
relation. The default positive/negative reference mobilities, 1.4/1.9 × 10⁻⁴
m²/(V s), are illustrative configuration values, not fitted measurements for
all environments. Ion masses remain 150/90 u. `ion_radius` (default 0.25 nm)
sets contact geometry independently of mobility. Aerosol friction retains
Stokes–Cunningham transport.

The transition interpolation uses the hard-sphere dimensionless fit with
Coulomb continuum/ballistic enhancement. With `a = r_a + r_i`, reduced mass
`m`, reduced friction `f`, and enhancement factors `Gc`, `Gk`, its normalized
Knudsen number is `sqrt(kBT*m)/(f*a) * Gc/Gk`. This yields the independent
limits `4*pi*a*kBT/f * Gc` and `pi*a**2*sqrt(8*kBT/(pi*m)) * Gk`.
The same normalization correction applies to the charged aerosol CPU/GPU
kernels; previous CPU/GPU parity alone did not establish these physical limits.

This approximation omits image forces, three-body trapping, ion hydration and
composition changes. Strong attractive collisions can therefore be
underestimated. It is not an extrapolation of LF2013 or an empirical model of
the whole atmosphere. Recombination remains the separately supplied constant
`recombination` in m³/s; varying attachment with T/P does not predict its T/P
dependence. Source, neutral cleanup and condensation exclusion remain explicit
companion processes. Check mass including injected/exported material and total
aerosol-plus-ion charge, rather than aerosol charge alone.

References: [Gopalakrishnan and Hogan (2012)](https://doi.org/10.1103/PhysRevE.85.026410)
for the transition scaling and Coulomb limits;
[Stow et al. (2017)](https://doi.org/10.1021/acs.analchem.7b01729) for the
low-field mobility relation. Constant cross section and fixed ion composition
are assumptions of this implementation.

See [custom processes across CPU and CUDA](../../examples/gpu/prepared/custom_process_backends.py)
for equivalent NumPy/Warp updates, explicit CPU transfers, a registered field,
absolute-time capture, detached snapshots and exact restart checks.

### Coagulation work diagnostics

`prepared.coagulation_diagnostics()` synchronizes and returns per-box expected,
required and available trials, the resource maximum timestep in seconds, and
actual scheduled trials, attempted trials and accepted collisions. Counts refer
to the latest step; fresh or resumed owners report zero until their first step.
Capture recording does not advance these counts. This compact observation does
not download particle arrays. The resource timestep is not an accuracy bound.

## Prepared dielectric-sphere coagulation

Prepare a `Coagulation` runnable with `VanDerWaalsCoagulationStrategy`
on a particle-resolved population. Preparation freezes the Hamaker constant,
static dielectric constant and geometry table version in the restart identity.
Graph replay evaluates live radii, environment and signed charges, including
charges produced by earlier merges. The existing equal-count and work-admission
rules apply to all boxes. [CUDA restart example](../../examples/gpu/prepared/van_der_waals.py)
