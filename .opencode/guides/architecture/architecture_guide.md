# Architecture Guide

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

### GPU nucleation staging boundary

- `particula.gpu.kernels.nucleation` is an unexported E6-F8 P1/P2/P3 seam,
  not a direct GPU step or runnable. P1 performs read-only validation; P2
  plans and inventory-admits source demand; and private P3 stages metadata for
  a later capacity-policy and activation phase.
- P3 converts P2-admitted demand times particle-box volume only when the
  binary64 result is finite, nonnegative, integral, and within the inclusive
  `int32` range. It retains the full provisional count even when it exceeds
  current free capacity.
- P3 delegates active/free classification and ascending free-index ordering to
  concrete-only E6-F5 `get_slot_diagnostics_gpu`. It writes only its supplied
  count, selected-index, and E6-F5 diagnostic sidecars; the selected prefix is
  limited by free capacity and unused index lanes are `-1`.
- Conversion rejection occurs before any caller-owned P3 or E6-F5 sidecar
  write. E6-F5 preserves its diagnostic outputs on pre-launch validation
  failure. Following a successful asynchronous diagnostic or P3 commit launch,
  rollback is not promised and callers must synchronize before reading outputs.
- This seam has no package export, hidden transfer, CPU fallback, activation,
  E6-F6 exhaustion-policy resolution, particle/gas mutation, resizing, or
  integrated direct-GPU execution. Those responsibilities remain deferred to
  later phases.
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
