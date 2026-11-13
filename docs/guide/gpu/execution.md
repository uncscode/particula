# Execution

## Execution selection

The package-level selection and policy values are stable and dependency-neutral.
Their complete availability and explicit-fallback contract is in [Backend
Selection and Explicit CPU Fallback](backend-selection.md). The
`particula.gpu` containers, conversions, and direct kernels are supported
experimental APIs with explicit caller-owned transfer and synchronization.

```python
from particula import (
    Backend,
    Capability,
    CapabilityDeclaration,
    CapabilityMatrix,
    CapabilityRequirements,
    Device,
    ExecutionAdapter,
    ExecutionContext,
    ExecutionRequest,
    Process,
)
```

Declarations and requests are immutable exact metadata. Nonempty requirements
must match a complete declaration exactly. Empty requirements are accepted when
the matrix contains a declaration for the same `Device` and `Process`.
`CapabilityMatrix.require()` completes this validation before
`ExecutionContext.resolve()` performs one exact context-local `(Process,
Backend)` lookup. The canonical CPU device spelling is exactly
`Device(Backend.CPU, "cpu")`.

The following CPU-only example demonstrates explicit construction, public
registration, and selection. Its `LocalAdapter.execute()` argument and return
behavior are local to this example, not a generic `ExecutionAdapter`, result,
state, identity, or mutation guarantee.

```python
from particula import (
    Backend,
    CapabilityDeclaration,
    CapabilityMatrix,
    CapabilityRequirements,
    Device,
    ExecutionContext,
    ExecutionRequest,
    Process,
)


class LocalAdapter:
    """Example-local adapter contract."""

    def execute(self, message: str) -> str:
        """Return this example's message unchanged."""
        return message


device = Device(Backend.CPU, "cpu")
process = Process("condensation")
requirements = CapabilityRequirements(frozenset())
matrix = CapabilityMatrix(
    frozenset(
        {CapabilityDeclaration(device, process, requirements)}
    )
)
context = ExecutionContext(matrix)
adapter = LocalAdapter()
context.register_adapter(process, Backend.CPU, adapter)
request = ExecutionRequest(Backend.CPU, device, process, requirements)

selected = context.resolve(request)
assert selected is adapter
assert selected.execute("example-local result") == "example-local result"
```

`selected is adapter` proves selection preserves adapter identity. No adapter
`execute` call occurs before the final assertion, which the caller invokes.
Registration and resolution neither execute an adapter nor mutate simulation
state. Invocation, arguments, state/result ownership, mutation declaration,
and runtime errors remain caller and registered-adapter responsibilities until
a later public contract exists.

An unsupported declaration fails before lookup with `ValueError`. A supported
declaration without an exact registration raises `LookupError`; no alternate
backend is selected.
Selection has ended after an adapter is returned and does not catch, retry, or
fallback from adapter failures.

### Selected-condensation ownership and API boundary

This is a non-runnable API and ownership reference. `particula.execution`
provides this frozen ordered 26-name public surface, also re-exported by
top-level `particula` alongside its unrelated APIs:

1. Selection declarations: `Backend`, `Device`, `Process`, `Capability`,
   `CapabilityRequirements`, `CapabilityDeclaration`, `CapabilityMatrix`,
   `ExecutionRequest`, `ExecutionAdapter`, and `ExecutionContext`.
2. Capability errors: `ExecutionCapabilityReason`,
   `ExecutionCapabilityError`, `UnknownExecutionTargetError`,
   `UnavailableExecutionTargetError`, `UnsupportedExecutionRequestError`,
   `UnknownBackendError`, `UnknownDeviceError`, `UnavailableRuntimeError`,
   `UnavailableDeviceError`, `UnsupportedProcessError`,
   `UnsupportedCapabilityError`, `InvalidExecutionStateError`, and
   `FallbackDisallowedError`.
3. Fallback policy enums: `FallbackPolicy`, `FallbackBoundary`, and
   `CPUStateAuthority`.

The concrete `particula.execution.errors` and
`particula.execution.fallback` modules remain direct-import-only; only the
listed error types and fallback policy enums are public. Fallback mechanics and
carriers, plus selected-condensation state carriers and adapters, are
intentionally concrete-only at
`particula.execution.adapters.condensation`; there is no provisional public
selected-condensation import, backend-registration recipe, or selected-step
workflow. Availability and fallback mechanics, adapters, resident seams,
registries, and GPU sidecars remain concrete-only.

Selected CPU execution retains a caller-owned native `Aerosol` and calls a
caller-owned `MassCondensation`. Selected Warp execution retains supplied
caller-owned resident `WarpParticleData`, `WarpGasData`, and
`WarpEnvironmentData`, plus same-device fixed-shape sidecars. The CPU profile is catalogue metadata and does not bind the supplied runnable
to its declared semantics. The selected Warp adapter checks required latent-heat
sidecars and activity/surface mode selectors against its profile before kernel
resolution. Native tagged state additionally requires a species binding pinned
to the exact particle/gas arrays; gas and particle species widths may differ. The result retains selected
state and native payload identity; it does not convert or restore either state
form.

Before resident Warp execution, callers explicitly use
`to_warp_particle_data`, `to_warp_gas_data`, and `to_warp_environment_data`.
Callers synchronize before host observation and restore only at their
checkpoint. Active resident state cannot fall back to CPU without a
caller-owned checkpoint/finalize snapshot and an explicit restore boundary.
Checkpoint, finalization, and restart are exact-device lifecycle behavior, not
a CPU fallback facility; the fallback module neither verifies nor performs
`RESTORED` recovery. Adapter dispatch does no hidden upload, restore, retry, or
silent CPU fallback. Direct-kernel validation may perform permitted device
scans or status readbacks, and omitted optional scratch or output buffers may
use direct-kernel fallback allocation; this allocation wording is unrelated to
backend fallback.

Successful Warp calls mutate particle masses, gas concentration, derived
GPU-only vapor pressure, finalized total transfer, eligible scratch/work
buffers, and optional energy output in place. `energy_transfer` is a
caller-owned write-only output, never a third return item; `thermal_work` is
validated deferred state.

Invalid adapter state, time, or profile rejection, and aggregate direct-kernel
preflight, preserve caller primary and output state before a writer launch. A
raw-proposal failure occurs before numerical validation commit in its failing substep, but
completed earlier substeps remain committed and documented vapor-pressure/work
sidecars may already be written. Callers needing retry semantics must use their
own snapshot/restore.

The [direct-kernel quick start](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_direct_kernels_quick_start.py)
is the supported explicit-transfer path. It is separate from the bounded
[GPU-resident session example](https://github.com/Gorkowski/particula/blob/main/docs/examples/gpu/resident/gpu_resident_session.py), which
documents lifecycle ownership rather than process execution.

### GPU-resident session lifecycle

`particula.execution` is the resident-session boundary, but its frozen 26-name
public export list excludes resident seams. Import resident seams only from their concrete
modules: `setup_resident_session`, `ResidentSession`, and `ResidentStepGuard`
from `particula.execution.gpu_session`; `GPUResourceRegistry` and its manifests
or views from `particula.execution.gpu_resources`; and checkpoint records,
`ResidentCheckpointController`, and `restart_resident_session` from
`particula.execution.checkpoint`. None of these names is a package or top-level
export.

`setup_resident_session` converts CPU particle, gas, then environment data once
and retains the resulting resident container identities. Dimensions are
immutable. Ordered gas names remain CPU-only metadata, while the registry pins
one exact active session and owns fixed-shape, same-device, nonaliasing sidecar
families. Registry acquisition validates or allocates resources only: it does
not execute, transfer, synchronize, or choose an RNG policy.

Resident stochastic state is concrete-only and session/registry-owned. Each
published process has one persistent, nonaliasing `(n_boxes,)` `wp.uint32`
sidecar: the separate `coagulation` and `wall_loss` namespaces never share one.
A stream identity is its schema-versioned stable logical box ID, root seed, and
namespace, not a physical lane, registry order, physical layout, or the
presence, activity, removal, or reordering of unrelated boxes. A frozen
configuration applies to covered enabled boxes; unrelated added,
disabled/no-work, removed, or reordered boxes do not alter a covered stream's
identity on the same supported device. This is not a cross-device replay claim.

This differs from direct kernels. Omitted direct-kernel RNG state is call-local
convenience state, while supplied state remains caller-owned and only
`initialize_rng=True` resets it. Resident first acquisition initializes a newly
published stream once; scheduled dispatch uses `initialize_rng=False`. Repeating
a resident root seed does not reset a stream. Only concrete-only
`inspect_streams`, `initialize_streams`, and `reset_streams` lifecycle calls can
inspect frozen metadata or explicitly initialize or reset all or selected
streams, and require the exact ACTIVE session/registry/closed-guard binding.
Inspection exposes no current arrays or words and performs no readback or
synchronization.

Normal resident steps do not transfer, read back, synchronize, reseed, restart,
fall back, or promise rollback. Checkpointing is the separate explicit boundary
for continuation; see [GPU resident checkpoints](resident/checkpoints.md).

A guard permits one token at a time. Checkpoint, finalization, close, and
discard require the exact session/registry/guard binding and a closed guard.
Normal guard bookkeeping neither bulk-transfers nor synchronizes. A read-only
failure can release its token and leave an ACTIVE session reusable; a possible
post-launch writer failure can fault the session without rollback. Close and
discard do not checkpoint, synchronize, restore, or mutate resident payloads.

A normal checkpoint is fresh and nonterminal; finalization terminalizes its
source session while returning an `ACTIVE`, restartable checkpoint. A checkpoint
owns immutable in-memory canonical bytes for primary arrays and acquired
sidecars, plus detached CPU inspection carriers. Inspection `GasData` is lossy:
it cannot recreate GPU vapor pressure or per-box partitioning. Canonical bytes,
rather than inspection carriers, are restart authority.
Snapshotting performs one synchronization and three `sync=False` inspection
restores, and needs roughly one additional host copy of resident payload bytes
as well as detached inspection copies. The first successful finalization caches
its snapshot; a later exact matching finalization returns that same object.

The implemented compatibility boundary is exact and fail-closed: restart accepts
a `ResidentCheckpoint` with carrier type `"ResidentSession"`, lifecycle
`ACTIVE`, complete valid payload descriptors and payloads, and an exactly equal
target `Device`. Only schema-v5 is accepted. It retains explicit population
semantics and independent gas and particle widths, optional complete closed-map
communication, and required RNG continuation metadata/current published words.
Schemas v1–v4 reject before setup or upload; no migration or inferred population
meaning is provided. Malformed, incomplete, mixed, or mismatched records,
non-`ACTIVE` checkpoint records, and device mismatches reject.
Restart is explicit, never automatic, and creates fresh session, registry,
guard, containers, primary arrays, and sidecars. It does not select or migrate
a device, provide fallback, or guarantee rollback after an asynchronous writer
launches. See
[GPU resident checkpoints](resident/checkpoints.md).

### Resident native graph-capture validation (test-only)

Resident native graph capture remains a concrete-only direct-import boundary,
not a public workflow. The delivered regression validation matrix uses CPU as a test-local NumPy
reference, Warp CPU as uncaptured evidence, and CUDA native capture as optional
pass-or-clean-skip evidence. Capture freezes the first physical timestep, and
each launch executes one timestep through one-token replay with authentic
opaque-handle provenance.

Opaque handles are privately owned and never exposed by captured records. One
LIVE provenance record admits replay through a per-record lease; teardown marks
it RELEASING before waiting and retains a RELEASED tombstone after its one
release callback. Native launch, abort, and release callbacks run outside the
global provenance lock. The session lifecycle protocol publishes terminal
intent before close, discard, or finalize callbacks, preventing callback
reentry and capture/replay races. Failures after capture begins abort native
capture, fault resident and graph state, and preserve the initiating exception.

The bounded regression evidence uses genuine nonzero native CUDA capture and
replay with exact READY bindings for single- and multi-box loops. It compares
meaningful particle, gas, environment, diagnostics, and partitioning fields
against both the independent NumPy oracle and uncaptured Warp CPU. Deterministic
parity, independent conservation checks, persistent RNG continuation, and
aggregate stochastic criteria remain separate; cross-device RNG words and
per-seed trajectories are not parity requirements.

Exact READY session/registry/closed-guard/request/publication binding,
qualification, explicit invalidation, retirement, renewal, and teardown remain
required. There are no public exports, no runnable example, no automatic
recapture, no fallback (including Warp-CPU capture), recovery, retry, rollback,
no hidden transfer/readback/synchronization, no checkpointed handle, or
cross-device replay.
This boundary makes no performance, profiling, or memory claim.

### GPU-resident deterministic timestep

#### Resident prepared enqueue boundary

`particula.execution.resident_enqueue` and
`particula.execution.resident_scheduler` are concrete-only direct-import seams;
prepared records and enqueue helpers are not `particula.execution` or top-level
imports. READY setup validates, normalizes, performs established private fallback
allocation, binds exact identities, and freezes control and operation selection.
Private prepared bindings then retain references and issue only frozen device or
write-free no-op callables: they perform no payload validation or readback,
allocation, transfer, synchronization, resource acquisition/replacement, device
selection/fallback, dynamic policy or schedule resolution, mutable-carrier lookup or
rebinding, or RNG initialization/reset. Direct wrappers may retain their
established bounded device-status observation after a prepared call; resident
prepared bindings are observation-free.

`prepare_resident_timestep()` retains an exact READY session, registry, guard,
request, signature, primaries, views, schedule, and duration by identity, but
does not authorize capture, token entry, or dispatch. The composed
`enqueue_prepared_resident_simulation()` is separate from an individual binding:
it first applies its retained structural/lifecycle identity gate before token
entry and a fresh signature comparison, opens exactly one token only after that
gate, and dispatches the twelve
frozen operations in canonical order. Empty or no-work bindings are write-free;
live pinned payload changes are permitted, while structural identities and frozen
scalar controls require a new prepared record. Capture capability owns lifecycle,
compatibility, CAPTURED admission, invalidation, and FAULTED classification; the
registry owns sidecars. A writer-capable failure retains the no-rollback, no-retry,
and no-fallback recovery boundary. Resource preallocation owns resource work; capture and replay native
capture/replay and captured parity remain downstream; captured-loop validation, benchmark collection, and CUDA profiling own later
parity/performance/memory/profiling evidence; graph-capture documentation owns the example,
limitations, and closeout.

`particula.execution.resident_scheduler` and
`particula.execution.diagnostics` are concrete-only direct-import seams. They
are not exported by `particula.execution` or the top-level package. The
scheduler accepts one identity-bound active session, pinned registry, closed
guard, and resolver-produced complete schedule; direct adapters retain the
resident objects by identity. Diagnostics accepts either the legacy ordered
two-snapshot plan (gas concentration, then saturation ratio) or the current
ordered six-operation plan, which appends total species mass, particle number,
latent energy, and conservation residual. Other subsets and orderings fail
closed. This migration rule retains old callers without creating a callback API
or implicit host inspection or transfer.

The closed schedule contains exactly these twelve node IDs:

`communication`, `volume_evolution`, `environment_update`, `gas_update`,
`vapor_pressure_refresh`, `saturation_refresh`, `condensation`,
`brownian_coagulation`, `dilution`, `wall_loss`, `nucleation`, and
`diagnostics`.

The resolver-produced canonical order begins with closed-map communication,
then optional prescribed volume evolution. These barriers use pre-update
volumes and invalidate saturation only; neither refreshes derived state. The
remaining order is profile/graph-dependent. Authoritative process dependencies
come from the resolved graph; virtual freshness edges are
`environment_update -> vapor_pressure_refresh`,
`environment_update -> saturation_refresh`, `gas_update -> saturation_refresh`,
`vapor_pressure_refresh -> saturation_refresh`, and
`saturation_refresh -> {condensation, diagnostics}`. The `SchedulerProfile`
selects exactly one reviewed edge: `condensation -> nucleation` or
`nucleation -> condensation`; the latter also requires nucleation before the
saturation-refresh consumer window.

| Resident responsibility | Authority and boundary |
| --- | --- |
| Closed-map communication and optional volume evolution | Concrete resident barriers using pre-update volumes; saturation invalidation only |
| Environment and gas updates | Prescribed identity-preserving resident updates |
| Vapor pressure and saturation | Derived virtual freshness state |
| Condensation, Brownian coagulation, dilution, wall loss, nucleation | Direct resident adapters over the resolved process nodes |
| Gas-concentration and saturation-ratio diagnostics | Closed snapshots to separate caller-owned outputs |

### Prescribed communication and volume evolution

Communication and volume evolution ships fixed-shape, single-device prescribed-map boundaries, not CFD or a
user-facing transport API. Direct gas communication accepts a GAS map with
closed in-domain edges and declared `-1` source/sink endpoints. Direct particle
transport accepts only closed in-domain, explicitly particle-resolved,
fixed-slot PARTICLES maps. Particle transport and physical volume evolution
reject missing/untagged, PMF and PDF population metadata before payload access. In contrast, a resident session pins exactly one complete
closed-map GAS or PARTICLES family and rejects combined or open forms.

Direct callers own the same-device fixed-shape primaries, maps, and work
storage. Gas communication uses caller-owned `(B, S)` `amounts`,
`amount_deltas`, and `outbound_amounts` `wp.float64` ledgers; a `(B, S)` source
or sink ledger is required only when its matching open endpoint is enabled.
Particle transport uses `(B, N)` debit/credit ledgers and `(E, N)` assignment
and request buffers. Optional final volumes are caller-owned `(B,)`
`wp.float64` values. Successful direct calls are asynchronous, so callers must
synchronize before inspection.

Direct gas stages immutable pre-step `amount = concentration * volume`, makes
one gas commit, and uses `new_concentration = final_amount / new_volume`. At
this standalone boundary, `new_volume` is the current unchanged volume, so its
sole normalization is equivalently `(amounts + amount_deltas) / volume`; it has
no fused direct-gas `new_volume` argument. In a resident schedule,
communication first stages with old volumes, then optional volume evolution
rescales gas concentrations by `old_volume / final_volume` and leaves particle
counts unchanged. Particle transport debits and credits the same count across
unequal-volume boxes; number density is count divided by the current volume.
This is not a second direct-gas normalization.

Closed gas maps conserve extensive species amounts within floating-point
tolerance. Open gas inventory changes are intentional and auditable through
the source/sink ledgers. Closed particle maps conserve total particle count,
every count-weighted mass lane, and count-weighted signed charge. Particle planning reads
immutable pre-step state, prefers an exact population match, then uses the next
ascending pre-step free slot. Capacity is deterministically gated: there is no
resize, compaction, implicit activation, or clipping.

Host/schema preflight rejects without primary or buffer mutation. During device
planning, documented work or planning buffers may change while primaries remain
gated; rollback is not promised after a writer launches. At the standalone
direct-kernel boundaries, validated empty or disabled maps and unchanged final
volumes are successful write-free no-ops. Resident composition has its own
barrier validation and does not make this general no-op guarantee.
Invalid map, schema, domain, alias, or device inputs, aggregate gas overdraw,
and required-but-missing open ledgers reject before primary commit.

The resident communication-then-volume barrier retains the twelve node IDs and
canonical order above: it uses old-volume staging, invalidates saturation only,
and retains fresh vapor pressure. Restart remains explicit and exact-device with
fresh identities. This scope excludes CFD, adaptive/distributed/multi-GPU
transport, hidden transfer or synchronization, CPU fallback, resizing or
compaction, resident RNG and restart RNG policy, graph capture, performance claims, and autodiff.

The direct scheduler skips virtual refresh-node dispatch. Instead, the
thermodynamic coordinator refreshes stale state in vapor-pressure-then-
saturation order immediately before condensation or diagnostics consumer
callbacks. Environment invalidates vapor pressure and saturation; gas,
communication, volume evolution, condensation, and nucleation invalidate
saturation. `volume_evolution_step_gpu` is independently callable at its
standalone direct-kernel boundary. Its use in resident composition is the
optional scheduled barrier; transport, mixing, and advection remain outside
this boundary.

Normal steps do not convert or transfer data, synchronize, restore, checkpoint,
finalize, restart, acquire or replace resources, fall back to CPU, retry, resize,
compact, capture graphs, claim performance, use multiple GPUs, or rewrite
physics. Preflight rejection leaves active state unchanged. After writer-capable
dispatch begins, a failure may fault the session and has no rollback or retry
guarantee. Checkpoint/finalize/restart remain the explicit exact-device boundary
documented in [GPU resident checkpoints](resident/checkpoints.md).

### Resident support validation matrix (2026-08-11)

The supported resident example is the repository-relative
[multi-timestep resident source](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/resident/gpu_resident_multi_timestep.py).
This closeout is scoped to [issue #1451](https://github.com/Gorkowski/particula/issues/1451).

**Status: closeout validation pending.** The historical four-module evidence
passed, but shipment now requires the corrected five-module coverage command
below to complete and record its literal result. Epic H and Epic I remain
deferred.

The diagnostics--example changed executable-module coverage target list is:

- `particula/execution/diagnostics.py`
- `particula/execution/gpu_resources.py`
- `particula/execution/checkpoint.py`
- `particula/execution/resident_communication.py`
- `particula/execution/resident_scheduler.py`

Documentation-only support publication files are not executable coverage targets. The required
changed-module aggregate gate is **at least 80%**. The historical four-module
closeout passed at 95%; the updated five-module command below must retain its
literal aggregate and per-target term-missing rows.

| Evidence category | Required command group | Outcome |
| --- | --- | --- |
| Selection/parity; lifecycle/identity; canonical scheduling/freshness; diagnostic ownership; closed-map transport and conservation; stream/checkpoint continuation; errors/no fallback; example/docs | Focused assertions | Passed: 289 assertions, 289 passed; coverage disabled. |
| Export boundary | Frozen export assertions | Passed: 15 assertions, 15 passed; coverage disabled. |
| Resident fast suite | Full resident suite | Passed: 891 tests, 891 passed; coverage disabled. |
| Full-package coverage | Whole-package coverage | Passed: 6,254 tests, 9 skipped, 93% coverage. |
| diagnostics--example changed-module coverage | Five-target coverage gate, `>=80%` aggregate | Run the command below and retain its literal aggregate and per-target term-missing rows, including `resident_communication.py`. |
| Strict documentation | Strict MkDocs build | Passed: `mkdocs build --strict` validated successfully. |
| Optional CUDA | CUDA guarded suites | Passed: `multi_box_loop_test.py` row and `condensation_integration_test.py` / `coagulation_integration_test.py` rows passed under `-m "warp and cuda"`. |

Run the groups sequentially; retain literal output, Warp availability, qualified
devices, test counts, coverage results, and per-target term-missing rows:

```bash
pytest particula/execution/tests/diagnostics_test.py \
  particula/execution/tests/gpu_resources_test.py \
  particula/execution/tests/checkpoint_test.py \
  particula/execution/tests/rng_invariance_test.py \
  particula/execution/tests/full_loop_test.py \
  particula/execution/tests/multi_box_loop_test.py \
  particula/execution/tests/transport_loop_test.py \
  particula/execution/tests/restart_loop_test.py \
  particula/execution/tests/condensation_integration_test.py \
  particula/execution/tests/coagulation_integration_test.py \
  particula/execution/tests/errors_test.py \
  particula/execution/tests/fallback_test.py \
  particula/execution/tests/fallback_integration_test.py \
  examples_tests/gpu_resident_multi_timestep_docs_test.py -q
pytest particula/execution/tests/exports_test.py \
  particula/tests/execution_exports_test.py -q
pytest particula/execution/tests/ -q
pytest --cov=particula --cov-report=term-missing
pytest particula/execution/tests/ -q \
  --cov=particula.execution.diagnostics \
  --cov=particula.execution.gpu_resources \
  --cov=particula.execution.checkpoint \
  --cov=particula.execution.resident_communication \
  --cov=particula.execution.resident_scheduler \
  --cov-report=term-missing --cov-fail-under=80
mkdocs build --strict
pytest particula/execution/tests/multi_box_loop_test.py -q \
  -m "warp and cuda"
pytest particula/execution/tests/condensation_integration_test.py \
  particula/execution/tests/coagulation_integration_test.py -q \
  -m "warp and cuda"
```

Acceptance classes remain separate: deterministic comparisons use their owning
explicit `rtol`/`atol`; closed concentration-weighted inventory and ledger
checks use `rtol=1e-12, atol=1e-30`; and stochastic evidence is parity validation's 100-seed
Warp-CPU aggregate/binomial evidence with an optional 12-seed CUDA bounded
smoke—not exact per-seed or cross-backend replay.

The supported boundary remains concrete-only and caller-owned: setup uploads
each CPU container once; registry resources are identity-pinned; diagnostics
copy only to caller-owned outputs; schema-v5 current words, not lossy inspection
carriers, are continuation authority; manual in-memory restart requires the
exact device and creates fresh identities; and source finalization caches its
first snapshot. Export expansion, automatic restart, device selection or
migration, hidden transfer/synchronization, CPU fallback, retry or rollback
after writer launch, open or unsupported resident communication, graph capture,
performance/scaling claims, autodiff, multi-GPU/distributed work, and
cross-backend exact RNG replay remain deferred.

### Complete direct-process illustration

The [complete direct-process source](https://github.com/Gorkowski/particula/blob/main/docs/examples/gpu/direct/gpu_complete_process_sequence.py)
is an illustrative explicit-transfer five-call sequence, not a resident
scheduler or coordinator example. It imports the five direct
boundaries from `particula.gpu.kernels` in order: `condensation_step_gpu`,
`coagulation_step_gpu`, `dilution_step_gpu`, `wall_loss_step_gpu`, then
`nucleation_step_gpu`. CPU↔Warp conversion helpers remain under
`particula.gpu`: use `to_warp_particle_data`, `to_warp_gas_data`, and
`to_warp_environment_data` for one conversion per CPU container, select the
device explicitly, then synchronize and restore once at the final checkpoint.
Configuration and scratch sidecars remain concrete-module imports.
Caller-owned sidecars have stable fixed shapes; persistent coagulation and
wall-loss RNG is also caller-owned. There is no hidden transfer or CPU fallback,
high-level `Runnable`, backend selector, scheduler, resident production loop, or
transport API. See the [direct-process validation evidence](https://github.com/Gorkowski/particula/blob/main/particula/gpu/tests/process_sequence_test.py),
[example regression](https://github.com/Gorkowski/particula/blob/main/examples_tests/gpu_complete_process_sequence_example_test.py),
and [GPU process completeness inventory](support.md#support).

`WarpParticleData`, `WarpEnvironmentData`, and `WarpGasData` are exported from
`particula.gpu` only when Warp is available, so import them only behind an
optional Warp guard:

```python
from particula.gpu import WARP_AVAILABLE

if WARP_AVAILABLE:
    from particula.gpu import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )
```

## Explicit CPU fallback boundary

`particula.execution.fallback` is a direct-import-only mechanics boundary; the
module, its operations, and its request/result carriers are not exported from
`particula.execution` or top-level `particula`. Its three policy enums
(`FallbackPolicy`, `FallbackBoundary`, and `CPUStateAuthority`) are the
intentional exception and are public at both boundaries. A caller that receives
a typed capability error may construct a concrete `FallbackRequest` with the
default-deny `FallbackPolicy.RAISE`, which re-raises the same eligible error.
Only an explicit `FallbackPolicy.CPU` request may select the already-registered
CPU adapter, and only for `UNKNOWN_DEVICE`, `RUNTIME_UNAVAILABLE`,
`DEVICE_UNAVAILABLE`, `PROCESS_UNSUPPORTED`, or `CAPABILITY_UNSUPPORTED`.

The caller must provide exact `CPUExecutionState` at either the
CPU-authoritative `PRE_UPLOAD` boundary or the caller-asserted `RESTORED`
boundary. Resident, uploaded, and mutated state claims reject; this boundary
does not restore or verify restoration. A successful fallback wrapper exposes
ordered `requested_backend`, `selected_backend`, and `capability_reason`
provenance while leaving the native `ExecutionResult.metadata` unchanged.

This explicit seam provides no package export, implicit CPU fallback, movement,
conversion, synchronization, checkpoint/finalize/restore operation, retry, or
post-writer rollback.

