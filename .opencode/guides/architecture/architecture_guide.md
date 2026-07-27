# Architecture Guide

## Execution Capability and Selection Boundary

- `particula.execution` is a dependency-neutral, standard-library-only,
  explicit-selection seam. The package-level public APIs are exactly
  `Backend`, `Device`, `Process`, `Capability`, `CapabilityRequirements`,
  `CapabilityDeclaration`, `CapabilityMatrix`, `ExecutionRequest`,
  `ExecutionAdapter`, and `ExecutionContext`, plus the public
  `ExecutionContext.register_adapter()` method. The per-context backing
  registry remains private.
- Declarations and requests are immutable exact metadata. Nonempty requirements
  must match a complete declaration exactly; empty requirements are accepted
  when the matrix contains a declaration for the same `Device` and `Process`.
  CPU metadata is spelled exactly `Device(Backend.CPU, "cpu")`; native
  identifiers for other backends remain opaque. Registration and resolution are
  context-local, selection-only operations: complete exact matrix validation of
  nonempty requirements occurs before one exact `(Process, Backend)` registry
  lookup. A resolved adapter retains
  identity, and the adapter is not executed during selection.
- This seam does not import, probe, or resolve a backend; perform availability
  detection; transfer or synchronize state; execute an adapter; retry; or
  fallback. In particular, generic `ExecutionAdapter` argument, result, state,
  mutation, and runtime-error semantics are not public. P3/P4 state, result,
  mutation, and concrete CPU execution-adapter types remain direct-module-only.
- Strategy physics, builder configuration, and existing CPU `RunnableABC`
  behavior are separate from E7-F1 typed selection and downstream process
  adapter/session layers. The exact downstream ordering is
  `E7-F1 -> E7-F6 -> {E7-F2, E7-F3, E7-F4} -> E7-F5`: E7-F6 owns availability,
  fallback, error taxonomy, API stability, and export policy. GPU adapters,
  resident sessions or loops, schedulers, implicit transfer/synchronization,
  retry, fallback, and replacement of direct GPU APIs remain deferred.

## Concrete Condensation Execution Boundary

- `particula.execution.adapters.condensation` remains concrete-only. It retains
  P2 CPU/Warp resource carriers and selected P3 CPU/Warp execution-state and
  adapter types; none are promoted through `particula.execution`, its adapters
  package, or top-level `particula`.
- P3 carriers retain their exact P2 state, controls, and CPU runnable by
  identity. They are frozen against field rebinding, but retained caller-owned
  resources remain mutable. Construction is side-effect-free and P2 retains its
  read-only ownership and metadata validation boundary.
- The selected CPU adapter completes local validation, requires the selected
  isothermal CPU profile, then calls the supplied `MassCondensation.execute()`
  once with the original aerosol, `time_step`, and `sub_steps`. The runnable
  must return that same aerosol. The adapter neither splits controls nor catches
  delegate exceptions.
- The selected Warp adapter completes local validation and profile preflight
  before lazily resolving `condensation_step_gpu`; it then makes one direct
  native call with the retained resources and forwards its native result
  unchanged. It imports neither Warp nor `particula.gpu` on the CPU path, and
  does not resolve the direct kernel before successful Warp preflight.
- Both adapters report state mutation while retaining state and backend results
  by identity. They perform no conversion, allocation, transfer, restoration,
  synchronization, retry, fallback, or post-launch recovery. Backend exceptions
  propagate unchanged; a launched Warp kernel retains its native rollback
  limits.
- The CPU selected boundary remains isothermal and rejects semantic latent heat.
  The selected Warp boundary forwards caller-owned `latent_heat`,
  `energy_transfer`, and deferred `thermal_work` sidecars by identity. The
  direct kernel retains thermal-sidecar validation, execution, exception, and
  post-launch authority; the adapter adds no transfer, allocation,
  synchronization, fallback, result reconstruction, or rollback behavior.

## CPU Nucleation Boundaries

- `particula.dynamics.nucleation` provides the bounded CPU-only P4 construction
  API: immutable activation/kinetic potential-rate strategies,
  `NucleationSourceConfig`, their builders, and `NucleationFactory`. These
  names are deliberately re-exported through `particula.dynamics`.
- P4 strategies calculate potential formation-event rates only. They do not
  create particles, admit inventory or slots, mutate gas or particle state, or
  provide GPU integration.
- `particula.dynamics.Nucleation` and `NucleationCommitConfig` are the
  supported CPU-only, single-box P5 process boundary. The runnable takes public
  P4 source configuration and `EnvironmentData`, adapts the legacy `Aerosol`'s
  existing particle and partitioning-gas backing data by identity, and returns
  that same aerosol.
- P5 splits a positive duration into equal sequential substeps. Every substep
  reads the gas state produced by prior successful substeps before evaluating
  the potential rate, then delegates source-demand finalization and commit to
  the concrete P2/P3 boundary. Atomicity is per attempted P3 substep, not for
  the complete `Nucleation.execute` call; successful earlier substeps persist
  if a later substep fails.
- P2 source-demand and P3 transaction records/helpers remain deliberately
  concrete-only in `particula.dynamics.nucleation.particle_source`; they are
  not package exports and P4 construction types do not import them. In
  particular, public construction does not expose particle-source
  finalization or commit helpers.
- P5 does not introduce GPU execution, hidden data transfer, or additional
  runnable/scheduler orchestration.

## CPU Particle Slot Management Boundary

- `particula.particles.slot_management` owns CPU-only fixed-slot classification,
  discovery, and direct-import activation for `ParticleData`.
- `get_slot_diagnostics` is its sole package-level export through
  `particula.particles`; `activate_slots` remains a direct import from
  `particula.particles.slot_management`, and validation helpers remain
  module-private.
- Discovery preserves all `ParticleData` storage and returns newly allocated
  fixed-shape `int32` free-index and count sidecars. Activation maps request
  prefixes to ascending free slots after complete read-only preflight, then
  mutates only mass, concentration, and charge storage in place.
- Storage resize or compaction, `ParticleData` mutation API changes, CPU↔GPU
  transfer, GPU execution, and a top-level particles activation export remain
  outside this boundary. Its fixed-shape behavior provides a deterministic CPU
  reference for later parity work.

## GPU Module Boundaries

The GPU package keeps a strict separation between transfer, schema, and
kernel-entry responsibilities.

### Transfer boundary

- `particula/gpu/conversion.py` owns explicit CPU↔GPU transfer helpers only.
- It should not absorb launch-time kernel validation or normalization logic.

### Schema boundary

- `particula/gpu/warp_types.py` defines Warp-backed container schemas only.
- It should remain a passive data-shape layer rather than a behavior layer.

### Kernel normalization boundary

- `particula/gpu/kernels/environment.py` owns shared private normalization and
  validation for GPU kernel entry points.
- This module is the common boundary for accepting legacy scalars, direct
  `(n_boxes,)` Warp arrays, or `WarpEnvironmentData` inputs before launch-time
  work.
- Condensation and coagulation should reuse this boundary rather than
  re-implementing environment validation independently.

### GPU package export boundary

- `particula.gpu` remains the public home for Warp availability, context, and
  explicit CPU↔GPU transfer helpers.
- Direct GPU step entry points should be imported from
  `particula.gpu.kernels`, not re-exported from top-level `particula.gpu`.
- Lower-level kernel helpers should stay module-local to
  `particula.gpu.kernels.condensation` and
  `particula.gpu.kernels.coagulation` unless a broader public contract is
  intentionally documented.
- Import the supported low-level dilution entry point with
  `from particula.gpu.kernels import dilution_step_gpu`.
- `dilution_step_gpu` completes deterministic, read-only validation before
  allocating private storage, launching a kernel, or mutating caller-owned
  state. Successful calls update particle and gas concentrations in place as
  `c_new = c * exp(-alpha * time_step)` and return the identical containers.
- The preflight guarantee ends at launch: post-launch rollback is not
  provided. This direct entry point does not imply CPU fallback or runnable
  support.
- Import the supported fixed-slot activation boundary with
  `from particula.gpu.kernels import activate_slots_gpu`. Its P3
  `get_slot_diagnostics_gpu` helper remains concrete-module-only at
  `particula.gpu.kernels.slot_management` and must not be re-exported.
- `activate_slots_gpu` maps selected request-prefix ranks to ascending free
  slots in caller-owned, fixed-capacity Warp storage. It reads and writes only
  particle mass, concentration, and charge; density and volume are
  intentionally unobserved. Requests and all activation/diagnostic `int32`
  sidecars are caller-owned, same-device inputs and outputs.
- P4 validates schema, ownership, current slot state, selected requests, and
  capacity before launching its writer. Rejected calls make no caller mutation
  or hidden CPU↔GPU transfer; after a writer launches, rollback is not
  promised. This direct boundary does not establish resizing, compaction,
  hidden fallback, or a higher-level runnable API. See
  [ADR-002](decisions/ADR-002-gpu-fixed-slot-activation-boundary.md).

### Direct GPU nucleation boundary

- `particula.gpu.kernels.nucleation` concretely implements the supported
  low-level import `from particula.gpu.kernels import nucleation_step_gpu`.
  P1 preflights fixed-capacity state and caller-owned sidecars; P2 admits shared
  inventory-safe source demand; P3 stages fixed slots; P4 resolves exhaustion
  with resampling before scaling fallback; and fused P5 activates selected slots
  and commits the matching gas transfer.
- Only `nucleation_step_gpu` is package-exported. `NucleationConfig`, P2/P3
  records, exhaustion controls, sidecars, and helpers are concrete-only in
  `particula.gpu.kernels.nucleation`; they are not promoted through either GPU
  package namespace.
- The direct caller owns CPU↔Warp conversion, same-device contiguous fixed-shape
  sidecars, device placement, and synchronization before inspecting successful
  asynchronous results. The step returns the identical particle and gas
  containers and provides no hidden transfer or CPU fallback.
- P3 retains demand beyond free capacity and uses ascending free-slot prefixes
  with `-1` tails. P4 selects resampling only when it fully covers a deficit,
  then uses representative-volume scaling as the configured fallback. P5 owns
  the selected-slot activation and matching gas mutation.
- Public rejection before P4 primitive entry preserves particle and gas state;
  P2--P4 may have written their documented sidecars before a later rejection.
  No rollback is promised after an E6-F6 primitive has been entered or a P5
  writer has launched.
- This boundary has no resize/compaction, GPU `Runnable`, scheduler/backend
  integration, or E6-F9 integration. E6-F9 remains a downstream
  explicit-transfer consumer.
- Import the supported fixed-slot wall-loss boundary with
  `from particula.gpu.kernels import wall_loss_step_gpu`. Its
  `NeutralWallLossConfig` is deliberately concrete-module-only at
  `particula.gpu.kernels.wall_loss`; do not re-export it through
  `particula.gpu.kernels` or `particula.gpu`.
- `wall_loss_step_gpu` owns immutable host configuration and frozen preflight
  for particle-resolved neutral and charged inputs. It dispatches the
  unchanged neutral kernel for neutral mode; charged kernels compose the private
  image-charge and electric-field-drift helpers in
  `particula.gpu.dynamics.wall_loss_funcs` only for nonzero-charge slots.
  Image-charge enhancement remains active for nonzero charge at zero wall
  potential, while charged zero-charge slots retain the exact neutral
  coefficient and RNG path. Spherical charged execution preserves a signed
  scalar field before adding the signed potential-derived contribution.
  Rectangular execution reduces caller-owned `(3,)` `float64` Warp field
  storage to its Euclidean magnitude before adding the signed
  potential-derived contribution; component signs do not individually select
  drift direction. The rectangular field is passed only to the charged
  rectangular kernel.
- After successful preflight, a nonzero call stochastically clears eligible
  fixed slots in place and returns the identical particle object. Removed slots
  have every mass lane, concentration, and caller-owned `charge` cleared;
  capacity and unselected storage are preserved. Zero time is write-free;
  pre-launch failures are atomic; rollback after a mutation launch is not
  promised.
- Its caller-owned `WarpParticleData.charge` field and optional `(n_boxes,)`
  `uint32` RNG sidecar remain external state rather than hidden transfer
  results. Successful positive-time calls advance each box sequentially only
  for eligible slots, while omitted RNG state is private to the call. Explicit
  `initialize_rng=True` is the only supplied-state reset path. Zero-time,
  preflight failures, and positive-time inputs with no usable slots leave a
  supplied sidecar unchanged. This serial per-box ownership is a correctness
  constraint, not a throughput claim. Runnables, hidden transfers/fallbacks,
  and CPU/Warp stochastic parity remain deferred. See
  [ADR-001](decisions/ADR-001-neutral-gpu-wall-loss-boundary.md).

## Design Intent

- Keep CPU↔GPU transfers explicit.
- Keep Warp container definitions stable and behavior-free.
- Keep cross-entry-point normalization private to `particula/gpu/kernels/`.
- Share validation at kernel boundaries when multiple GPU entry points consume
  the same environment contract.
- Keep GPU exports deliberate: top-level helpers in `particula.gpu`, direct
  step entry points in `particula.gpu.kernels`.
