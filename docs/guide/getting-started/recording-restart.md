# Record, save and restart a simulation

Use this workflow to retain a physical-time history and continue a supported
calculation from its accepted state. Start with the
[first CPU experiment](first-simulation.md) for executable native state and
physics setup. These concrete `particula.execution` imports describe development
0.3.1; they do not change existing runnable calls or select physical assumptions
for you. Choose tolerances using the
[numerical-method guide](../processes/numerical-integration.md).

## Select an execution owner

| Owner | Advance and observe | Exact restart profile |
| --- | --- | --- |
| `CPUSimulation` | `run_until`; native CPU runnable cycles | Known native processes, CPU RNG, auxiliary fields and diagnostics |
| `CondensationODE` | `run_until`; CPU RK23 or ROS2 accepted steps | Scaled state, controller and supported forcing table |
| `GPUCondensationODE` | `run_until`; Warp CPU or CUDA RK23 | Same explicit device, scaled device state and controller |
| `SplitSimulation` | `run_until`; complete sequential or Strang cycles | Nested built-in condensation/dilution flows and their controls |
| `PreparedSimulation` | `run_prepared`; unchanged fixed step grid | Existing qualified prepared configuration and canonical resident checkpoint |

The CPU owner uses native NumPy execution without launching Warp. Adaptive
condensation supports the physical profiles in the numerical-method guide;
it does not add adaptive stochastic coagulation or latent-heat integration.
Prepared execution retains its existing supported stochastic and latent-heat
profiles. Custom deterministic split callbacks may run but cannot be saved as
exact restart configurations. Unknown CPU processes are rejected. Registered
GPU extensions require the caller's matching registry when resuming; an archive
never supplies executable adapters.

## Record selected variables at physical times

Given a configured `simulation`, request absolute times in seconds:

```python
from particula.execution.recording import Recorder, run_until

history = Recorder(
    ("number_density", "particle_mass_density", "gas_concentration"),
    max_records=1000,
    max_bytes=16 * 1024 * 1024,
)
run_until(simulation, 10.0, output_times=[0.0, 2.0, 5.0], recorder=history)
arrays = history.arrays()
assert arrays["time"][-1] == 10.0
```

The final time is always an output target. The initial time is recorded only
when requested. Times must be finite, nonnegative and strictly increasing;
requests older than the current accepted time are skipped when continuing.
Reuse the same recorder across calls to omit an equal-time joining row.
A new recorder starts a separate history. Layout or species changes require a
new recorder; matching array widths alone do not establish matching species.

`run_until` bounds steps at each requested time, without interpolation. Only
accepted complete steps or split cycles are visible, never RK stages, rejected
trials or Strang half cycles. Shortened steps can change a numerical or
stochastic trajectory. Exact continuation and passive-observer invariance
require the same future step/output schedule; compare different schedules
against an independent accuracy target or stochastic uncertainty.

| Variable | Unit | Axes after time |
| --- | --- | --- |
| `number_density` | m⁻³ | box |
| `particle_mass_density` | kg m⁻³ | box, material |
| `gas_concentration` | kg m⁻³ | box, gas |
| `temperature`, `pressure` | K, Pa | box |
| `volume` | m³ | box |
| `particle_masses` | kg per particle | box, slot, material |
| `particle_concentration` | native population measure | box, slot |
| `particle_charge` | elementary charges | box, slot |

Volume is always retained. Slot variables are opt-in: resolved concentration is
represented count, discrete concentration is number density, and a PDF is
*dN/dr*. Metadata retains dimensions, units, gas order and molar masses,
partitioning, material densities and population interpretation. Rows and
returned arrays are detached. The byte limit bounds retained numerical data,
including times; Python overhead and serialization copies require extra memory.
A capacity failure leaves accepted physics intact and appends no partial row.

## Save analysis output

`save_analysis(history, path, ...)` saves on demand. To save after each requested
observation, wrap the recorder in `AnalysisWriter` and pass it to `run_until`:

```python
from particula.execution.analysis_io import AnalysisWriter, load_analysis

writer = AnalysisWriter(
    history,
    "growth.analysis",
    configuration={"case": "my explicitly configured growth experiment"},
    numerical_settings={"output_times_s": [10.0, 15.0, 20.0]},
    source_identity={"study": "growth-study", "input_revision": "case-1"},
)
run_until(simulation, 20.0, output_times=[10.0, 15.0], recorder=writer)
result = load_analysis("growth.analysis")
assert result.arrays["time"][-1] == 20.0
```

Replace the illustrative provenance with the actual physical parameters,
process ordering, species maps, tolerances, source revision and input identities.
These dictionaries must contain finite JSON data; the writer cannot infer your
scientific assumptions. Package version and scientific array metadata are added
automatically. `compressed=False` selects uncompressed numeric payloads.

The versioned single-file ZIP/JSON/NPY format is intended for bounded histories.
It retains numeric arrays without pickle, checks hashes, duplicate members,
array headers and size budgets, and does not execute file-named code. It is not
a chunked database for unlimited particle trajectories. Analysis and restart
schemas are distinct: a history cannot masquerade as an exact checkpoint.

A save writes and flushes a temporary file before atomic replacement. If writing
fails, the previous complete file remains and the new in-memory row is retained.
After resolving the storage problem, call `writer.flush()` to retry without
advancing physics or duplicating that row. File failure does not roll physics
back. Use separate output paths for separate studies and keep the recorder under
one caller's control while recording or saving.

## Apply a reproducible environmental history

For CPU or GPU adaptive condensation, pass an immutable table as `forcing=`:

```python
from particula.execution.forcing import EnvironmentForcing

forcing = EnvironmentForcing(
    times=[0.0, 5.0, 10.0],
    temperature=[298.15, 303.15, 298.15],  # K
    pressure=101325.0,                    # Pa
    mode="step",
)
```

Times start at zero and increase strictly. Temperature and pressure accept a
scalar, one value per time, or a time-by-box matrix. A single column broadcasts
across boxes. Values must be positive and finite. `mode="linear"` interpolates
between rows; `mode="step"` holds the current row. The last row is held forever.
The table is copied into immutable data and is included in supported checkpoints.

Adaptive integration stops at every knot even when no output was requested
there. Endpoint stages finishing the previous segment use that segment's
left-hand environment; an accepted snapshot at the knot shows the new,
right-continuous environment. This prevents an RK endpoint stage from sampling
the next step's discontinuity. Built-in `condensation_flow(..., forcing=forcing)`
also makes the outer split cycle stop at knots. Refine the outer step separately
when checking splitting accuracy.

A pure callable forcing remains available for adaptive execution and evaluates
at absolute stage times. It must not consume RNG or mutate external state.
Its discontinuity boundaries and one-sided values remain the caller's
responsibility, and it has no supported exact disk identity. Prepared GPU
forcing instead uses its existing explicit environment-update interface;
`run_prepared` does not install this adaptive table automatically.

## Restart accepted CPU or adaptive GPU state

For an adaptive CPU `simulation` and its matching `strategy`:

```python
from particula.execution.cpu_restart import (
    checkpoint_condensation, load_cpu_checkpoint,
    resume_condensation, save_cpu_checkpoint,
)

save_cpu_checkpoint(checkpoint_condensation(simulation), "growth.restart")
resumed = resume_condensation(load_cpu_checkpoint("growth.restart"), strategy)
assert resumed.time == simulation.time
```

`CPUSimulation` instead supplies `simulation.checkpoint()` and resumes with
`CPUSimulation.resume(load_cpu_checkpoint(path), process)`. It retains the
accepted native state, process configuration, time/cycle count, timestep,
CPU RNG, supported extension fields and additive ledgers, and latent-energy
diagnostics. State and RNG from a failed current cycle are discarded together.

For `SplitSimulation`, use `checkpoint_split(simulation)` and
`resume_split(load_cpu_checkpoint(path), group)` from
`particula.execution.split_restart`, with the same CPU save function. The
caller-supplied group must match names, nesting, ordering, physics and controls.
The checkpoint is at a complete cycle; component controllers already restart
at each cycle, so there is no hidden half-step controller to recover.

For a GPU adaptive owner, use the distinct functions in
`particula.execution.gpu_restart`:

```python
from particula.execution.gpu_restart import (
    checkpoint_gpu_condensation, load_gpu_checkpoint,
    resume_gpu_condensation, save_gpu_checkpoint,
)

save_gpu_checkpoint(checkpoint_gpu_condensation(gpu), "gpu-growth.restart")
gpu_resumed = resume_gpu_condensation(
    load_gpu_checkpoint("gpu-growth.restart"), strategy, device="cuda:0",
)
assert gpu_resumed.time == gpu.time
```

Adaptive records retain exact scaled coordinates and tolerances, accepted time,
next proposed step, last error, work counters and forcing. Scratch stages are
recreated. GPU capture of a checkpoint explicitly downloads state; loading its
data alone does not allocate a device owner. Resuming uploads to the explicitly
selected, exactly compatible saved device, including CUDA UUID identity.
There is no cross-device continuation or hidden fallback.

CPU disk restart requires matching package and NumPy versions. GPU restart also
requires matching Warp and the existing device/configuration contract. Unsupported
schemas, changed physics, unknown records and incompatible devices reject;
there is no guessed migration of old records. Loaders decode known data records;
resume performs owner-specific configuration and state validation before use.
These strict compatibility checks do not replace retaining the exact code and
inputs of a research study, especially with a modified development checkout.

## Observe and restart prepared GPU runs

Given a supported `process` and native `aerosol`, explicitly choose downloads:

```python
from particula.execution.checkpoint_io import load_checkpoint, save_checkpoint
from particula.execution.prepared_recording import PreparedObserver, run_prepared

prepared_history = Recorder()
with process.prepare(aerosol, device="cuda:0", time_step=0.1) as prepared:
    observer = PreparedObserver(
        prepared, prepared_history, download="diagnostics",
    )
    run_prepared(
        prepared, 1.0, output_times=[0.0, 0.5], observer=observer,
    )
    save_checkpoint(prepared.checkpoint(), "prepared.restart")

with process.resume(load_checkpoint("prepared.restart"), device="cuda:0") as continued:
    observer = PreparedObserver(
        continued, prepared_history, download="diagnostics",
    )
    run_prepared(
        continued, 2.0, output_times=[1.0, 1.5], observer=observer,
    )
assert prepared_history.arrays()["time"][-1] == 2.0
```

Every output and final time must lie on the original fixed grid. Validation
rejects off-grid requests before stepping; it never changes the timestep to
reach an observation. Labels are computed from initial time plus integer step
count times timestep, with floating-point roundoff admission. Older requests are
skipped and equal-time joins do not download or append a duplicate observation.

Constructing `PreparedObserver` downloads one full snapshot to bind species and
layout metadata. `download="diagnostics"` then transfers existing per-box/species
reductions and environmental values, without full slot arrays. Requesting slot
variables requires `download="snapshot"`, which downloads the full native state
at each output. `initial_snapshot_downloads` and `observation_downloads` count
calls, not bytes. Adaptive GPU `run_until` always takes a full snapshot at each
new requested observation. Account for these transfers in workload timings.

A prepared observer may also wrap an `AnalysisWriter`. All host observation,
compression and serialization occur outside active CUDA capture. Qualified
captured replay can run between observations; no automatic capture, recapture
or CPU emulation is introduced. Restart retains canonical resident bytes,
managed RNG streams, required auxiliary fields and process continuation data;
CPU inspection carriers are not the restart authority. Registered extensions
require a matching explicitly supplied registry. See
[prepared execution](../gpu/prepared.md) and
[resident checkpoints](../gpu/resident/checkpoints.md) for their lifecycle and
configuration restrictions.

Preserve study outputs separately from published documentation bundles. The
[first CPU experiment](../../examples/fundamentals/first_cpu_workflow.py)
demonstrates plotting and an exact future-state check after restart; its
publication retains compact plot data rather than full particle states.
