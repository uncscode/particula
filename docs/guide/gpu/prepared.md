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
| Condensation | `MassCondensation` with isothermal or latent-heat strategy | Native discrete or resolved populations; explicit species maps; supported ideal/Kappa activity and constant/Water Buck vapor-pressure models; four fixed equal GPU substeps. |
| Coagulation | `Coagulation` with Brownian strategy | Particle-resolved, equal positive active counts, default interpolation. Other direct-kernel mechanisms are not prepared runnable capabilities. |
| Dilution | `Dilution` with exact exponential strategy | Counts/densities and gas inventory decrease consistently; no volume change. |
| Wall loss | `WallLoss` with spherical, rectangular or charged strategy | Particle-resolved; supported neutral/charged configuration is frozen at preparation; retired resolved slots retain the native raw-count semantics. |
| Nucleation | `Nucleation` with activation or kinetic strategy | Resolved unit-count, neutral injection with explicit species mapping, integral event demand and fixed capacity; supported exhaustion/resampling policy is retained. |

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
uses its own composition. CPU surface mixing and Kelvin terms preserve that
particle axis as well; independent rows are never averaged into one mixture.

Process configuration, timestep, box layout and buffer capacity are frozen.
Editing the original Python strategy after preparation does not change a running
simulation. Changing these requires a new preparation; a checkpoint resume also
requires matching configuration and the exact supported device.

## State, forcing and observations

`run(steps)` enqueues fixed-duration steps. `completed_steps` and
`simulated_time` track enqueued work; synchronize or observe to establish device
success. Ordinary prepared steps perform no simulation-state download or device
array allocation. Explicit observation and forcing boundaries wait for pending
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
