# Custom processes and registered fields across CPU and GPU

Accepted development plan, 2026-11-15. One branch:
`codex/process-field-extensions`; one PR to `uncscode/particula:main`.

## Required outcome

Expose a small public CPU extension contract: declare reads/writes, validate
requirements, advance one detached native single-box Aerosol using simulation
time and timestep, validate results, and publish after success. Users can call
arbitrary Python/scientific libraries. Value mutation does not authorize
species, shape, dtype, capacity, or array-identity changes. Scientific source,
sink, and conservation accounting belongs to the custom process.

Deliver registered fields, explicit GPU adapters, and opt-in CPU transfer
operations in the same PR. No implicit fallback, persistent CPU simulation
owner, arbitrary particle-event callbacks, or release/merge is authorized.

## Public contracts

- Export BoxProcess and FunctionProcess from particula. Public methods are
  `validate(aerosol)` and
  `advance(aerosol, *, time, time_step, context: BoxContext) -> None`.
- Declare reads, writes, representations, extension ID/version and unique
  instance name. Configuration is separate from mutable continuation state.
- BoxContext supplies logical box identity and requested transactional RNG.
  Random extensions require CPUExecutionContext. Key streams by instance and
  logical box; keep mutable continuation in registered state/managed RNG.
- Stage the whole batch per process substep; publish data and RNG only after
  every box passes. Earlier processes/substeps stay committed on later failure.
  External side effects are outside rollback. Detect undeclared writes and
  structural/metadata changes; do not treat trusted callbacks as sandboxed.
- Writable native fields: particle masses, concentrations, charge, valid volume;
  gas concentrations; temperature and pressure. Shared material/species metadata
  is read-only. Registered fields are writable when declared.
- Add keyword-only `time=0.0` to native and sequence CPU execution. Substep i
  starts at time + i*time_step/sub_steps; all processes share that interval.
  Callers supply start time on repeated calls. Validate zero-duration calls but
  skip custom advancement/RNG. Preserve legacy overrides through explicit
  dispatch, never by catching TypeError from arbitrary execution.
- Prepared owners accept initial time; current_time is absolute and existing
  simulated_time remains elapsed. Captured operations use a device clock.

## Registered storage

Export FieldSpec and ExtensionFields; Aerosol optionally owns a field store
accessed through `aerosol.fields[name]`. Descriptor: namespaced name, axes,
dtype, units, initial/birth value, validation bounds, particle-event policy.
Layouts: (box), (box, particle), (box, gas_species), (box, particle_species),
(box, particle, particle_species). Dtypes: float64/int32/int64; additive
tracers require float64. Units are metadata, not automatic conversion.

Fields are persistent state, distinct from derived caches and scratch. Integrate
copies, box views, alias validation, publication, replacement, snapshots,
transfers and checkpointing. Reject incompatible container replacement.
Prepared schemas and device identities are fixed; structure changes require
new preparation. A field does not automatically change physics or add species.

Particle fields initially support resolved populations:

- Passive: preserve identity, initialize births, clear retired slots; reject
  undefined mergers/resampling.
- Additive per particle: sum supported physical mergers, use the exact existing
  represented-count redistribution weights for resampling, retain per-particle
  values under count dilution, initialize births and clear retired slots.
  Check sum(n_i*q_i), and record built-in source/removal contributions.
- Integrate native CPU/GPU coagulation, activation, retirement, dilution and
  supported resampling. Reject representative-volume scaling with such fields.
- Custom processes preserve slot identity/activity when particle fields exist.
  Native built-ins implement population events. Low-level direct calls outside
  orchestration remain caller-managed. Inventory preservation is not proof of
  preservation of the joint tracer distribution.

## GPU composition and continuation

Use typed preparation contexts/plans/operation records instead of extension
helpers reaching through owner: Any. A simulation-local ProcessRegistry maps
exact types to adapters providing validation, immutable versioned plans,
preparation and retained enqueue operations. Explicitly declare resident,
capture, checkpoint, RNG and population-event support. Built-ins use the same
machinery without broadening scientific admission. Custom instances have unique
names; built-in family restrictions remain.

Contexts expose declared arrays, workspace, stream/device, clock, RNG and
failure reporting. Resident steps allocate/read back/synchronize nothing.
Freeze configuration fingerprints including extension version, instance,
parameters, declarations and schema; do not serialize Python code. Preserve
process order, declare invalidations, insert necessary existing thermodynamic
refreshes, and reject stale derived consumers without a refresh provider.

`CPUTransfer(process)` explicitly selects a host callback in a GPU sequence.
Unwrapped CPU-only processes reject. Drain and check the owner stream, download
full native/extension state, stage/run/validate CPU boxes at current time, copy
only permitted writes into existing device buffers, then refresh dependencies.
Reuse staging where practical. Never replace sessions/bindings or reseed RNG.
Callback rejection publishes no candidate state; earlier GPU writes remain.
Prepared-step failures fault without retry. Transfer sequences reject capture
before recording, but support checkpoint/resume.

Prepared checkpoint envelope v2 contains field descriptors/bytes, time,
ordered configuration/instance metadata, persistent adapter state, device and
host RNG, and execution mode. Keep resident schema unchanged unless its own
contract changes. Explicitly read v1 built-in-only checkpoints with zero start
time/no extensions. Resume requires caller-supplied compatible implementations
and registry on the supported exact device; reject drift before upload.
Capture signatures retain extension arrays, clock and adapter resources.
Recording advances neither scientific state, clock nor RNG.

## Serial commits and progress

- [x] docs(plan): record process and field extension design
- [x] refactor(execution): introduce typed process bindings
- [ ] feat(cpu): expose timed box process extensions
- [ ] feat(data): add registered extension fields
- [ ] feat(particles): preserve additive fields through population events
- [ ] feat(gpu): prepare registered processes and fields
- [ ] feat(gpu): add explicit CPU transfer processes
- [ ] feat(execution): checkpoint and capture extension state
- [ ] docs(examples): demonstrate process and field extensions
- [ ] test(execution): qualify extension workflows

Feature commits include focused tests, not just the final qualification commit.

## Examples and acceptance

Publish maintained percent-cell Python examples with executed saved bundles:

1. Two-box timed CPU gas chemistry using SciPy plus scheduled emissions through
   a function wrapper; native composition, analytical reference, inventory plot.
2. Additive particle tracer coagulation/removal plus a separate source/resampling
   case; inventory/distribution plots and passive-policy rejection. Preserve
   equal-count coagulation restrictions.
3. Equivalent time-dependent NumPy/Warp custom update; resident versus transfer
   execution, detached snapshots, restart and resident capture replay.

Use only public extension APIs, declare downloadable helper inputs, and test
extracted bundles outside the checkout. Update process/data/GPU guides,
docstrings, navigation and example registry. No root README changes.

Required evidence before completion:

- [ ] Single/multiple boxes, reordering, timing/substeps/repeated calls, zero
      duration, independent instances and transactional RNG.
- [ ] Requirement, schema, alias, undeclared-write, invalid-value and array
      replacement rejection, including later-box failure atomicity.
- [ ] Field copy/view/transfer isolation and independent additive inventories
      through slot reuse, merging, removal and exact resampling weights.
- [ ] Unextended native behavior remains numerically unchanged.
- [ ] Real CUDA resident execution has no per-step allocation/readback;
      transfers preserve device pointers and RNG and refresh dependencies.
- [ ] Failures suppress dependent work and surface at observation.
- [ ] Checkpoint bytes, time, host/device RNG, multiple instances, v1 compatibility
      and corrupted/incompatible rejection before upload.
- [ ] Real CUDA capture recording is inert; replay/restart matches uncaptured
      continuation; transfer capture rejects.
- [ ] Extracted examples, saved results, docs checks/site tests/build, Ruff,
      mypy, package/example/script suites and required coverage pass.
- [ ] Bounded extension-free overhead and resident/transfer timing evidence;
      distinguish Warp CPU from real CUDA, never claim unmeasured speedup.
- [ ] Verify final diff, current upstream, fork destination; push one branch and
      open one PR titled "feat: add custom processes and registered fields for
      CPU and GPU". Follow final-commit CI and live review threads to completion.

Use existing .venv and maintained validation commands. Record exact evidence
and limitations. Do not infer completion from implementation intent or a
narrow passing suite. Leave merge, version bump and release outside this goal.

## Implementation evidence

- Initial exact-type registry and immutable registration snapshots implemented;
  two focused tests pass. Typed binding integration remains pending.

- Prepared operation records now wrap built-in compiled operations and retain
  resource identities for capture traversal. Registry/operation tests and Ruff
  pass; mypy passes for extension_registry.py. Full typed context and built-in
  configuration migration is still pending, so the refactor is not complete.
- Initial restricted regression attempt: 58 passed, 69 skipped, 13 failed due
  to an unwritable Warp compiler cache. Host CUDA rerun uses temporary cache;
  running session 46417 at this checkpoint. Do not treat skips as CUDA evidence.
- Host regression session 46417 completed: 140 passed, no skips, including
  CUDA cases and prepared restart/capture. This qualifies the operation-record
  refactor only, not the remaining extension features.
- Initial public BoxProcess/FunctionProcess and explicit CPU sequence time are
  implemented. CPU contract remains under development: broaden multi-box/RNG,
  metadata/replacement tests, declaration validation and docs before completion.

- CPU contract tests now include later-box numeric/RNG rollback, duplicate
  instance rejection, metadata/container/undeclared-write rejection, and shared
  physical substep time. Focused CPU suites: 37 passed. Ruff and mypy for the
  new CPU module/runnable pass. Public process guide documents actual CPU API.
  Registered fields and GPU integration remain unimplemented at this point.
- Registered storage is connected to Aerosol, detached copies, box views,
  alias/publication checks and custom declarations. Storage/CPU/aerosol tests:
  43 passed. Native population-event integration remains pending.
- Native resolved CPU coagulation now passes the actual admitted merge pairs
  to additive field updates; passive policies reject before sampling. Remaining
  population work: birth/removal ledgers, resampling weights and GPU events.
- CPU additive fields follow exact coagulation pairs. Native dilution, wall
  loss and condensation retirement now clear freed slots and accumulate
  represented tracer removal in transactional per-box ledgers. Detached
  `inventory_changes(name)` reports source/removal arrays; ledgers copy and
  slice with field state and publish atomically. Source population ledger
  integration remains pending. Focused CPU/multi-box suite: 115 passed;
  field suite including independent dilution/removal budgets: 13 passed.
- CPU particle sources now stage registered fields with native state, preserve
  additive tracers using the exact resampling count-overlap records, initialize
  reused slots from field defaults, and record birth inventory. Passive fields
  reject resampling; particle fields reject representative-volume scaling.
  Late field-ledger overflow rolls back the entire source transaction. Native
  nucleation forwards its field store into this transaction.
- Source/resampling qualification: 227 tests passed across extension fields,
  source transactions, exhaustion, native runnables and CPU batching; Ruff and
  targeted mypy passed. GPU population events and public examples remain pending.
- Registered resident adapters now connect to public `prepare(process_registry=)`:
  immutable versioned GPUProcessPlan, narrow GPUPreparationContext, exact-type
  registry snapshots and retained operations. Non-particle fields upload once
  with their declared dtype and snapshots own detached values. Device time
  advances within the captured sequence; `time=` and `current_time` preserve
  absolute versus elapsed time. Tests cover explicit registration, bad planners,
  immutable writes, duplicate instances, detached configuration, field ownership,
  allocation/readback-free stepping and native CUDA capture with evolving time.
- Qualification: prepared/restart/registry/extension suites passed 153 tests
  (one intentional Warp-CPU capture skip). Added initial-time checks plus restart
  regression passed 61 tests (same intentional skip); targeted Ruff and mypy
  passed. This remains intermediate: schema-v2 checkpoints, field value guards,
  GPU particle events, RNG/adapter state, CPUTransfer, refresh invalidations and
  migration of built-in helpers to typed contexts remain pending. Until those
  implementations land, particle-field GPU execution and extension checkpoints
  reject explicitly rather than discard state.
- Prepared checkpoint schema v2 now stores canonical registered field bytes,
  descriptors, declared GPUStateSpec continuation arrays, absolute initial time,
  and the exact device clock. Saved process configurations exclude live adapter
  objects. Resume accepts caller-supplied exact-type registry implementations,
  verifies their configuration and the full payload inventory before upload,
  and restores into freshly prepared fixed buffers. An explicit v1 reader
  admits only legacy built-in-only state at zero origin. Resident schema stays 5.
- Checkpoint qualification: 170 prepared/extension/restart tests passed on Warp
  CPU and CUDA, with one intentional CPU capture skip. Coverage includes two
  independent custom instances, native captured replay before and after restart,
  exact field/state/clock bytes, corrupted byte/shape inventory and version
  rejection before allocation, and nonzero built-in initial time. Managed custom
  RNG and CPU-transfer host RNG continuation remain pending; the generic device
  state contract alone is not evidence those RNG requirements are complete.
- CPUTransfer now explicitly wraps public Python box processes in prepared
  sequences. It drains/checks earlier device work, downloads full state, stages
  and validates all CPU boxes, and copies only declared writes through reusable
  upload staging into retained device arrays. Transfer sequences reject capture.
  CPU process metadata and immutable parameters participate in restart matching;
  schema-v2 mode and canonical PCG64 host RNG bytes survive exact continuation.
- Thermodynamic composition refreshes after custom gas/temperature writers when
  condensation supplies the provider, and rejects dependent saturation readers
  without one (including across repeated timesteps). A native saturation-gated
  source participates in this check without restricting ungated source physics.
- Transfer qualification: affected prepared/extension/restart suites passed 187
  tests with one intentional CPU capture skip. Expanded transfer suite passed 19
  tests on Warp CPU and real CUDA, including a same-step saturation reader,
  multi-box callback failure atomicity, fixed device pointers, native GPU RNG
  preservation, host RNG restart and corrupt-host-state rejection before upload.
  Targeted Ruff and mypy pass. Device custom RNG, native GPU particle fields,
  field-value guards, typed built-in contexts, examples and final PR gates remain.
- Resident custom processes can now request managed uint32 Warp random words
  keyed by root seed, extension ID, instance and logical box. Preparation checks
  same-process word collisions; steps never reseed. Random bytes are reserved
  continuation state and participate in schema-v2 validation, exact restart and
  capture identities. Zero-duration resident custom calls skip advancement.
- Declared custom writes and floating adapter state now have prepared device
  validation launches. Finite/bounds failures latch code 90 and suppress later
  guarded work and clock advancement; observation faults the owner. Integer
  comparisons preserve int64 precision above 2**53. Guards allocate/read back
  nothing per step. GPU callbacks still own the required failure-flag check and
  scientific accounting; these checks are not a security boundary.
- Qualification: 201 affected CPU/CUDA tests passed with three intentional CPU
  capture skips. Expanded RNG/dtype qualification passed 17 tests with three
  CPU capture skips, covering float64/int32/int64 bounds, logical-box/process
  reordering, independent instances, inert capture and exact random continuation,
  missing RNG rejection before upload, and zero-duration preservation. Ruff and
  targeted mypy pass. Native GPU particle-field events and final deliverables
  remain pending.

- Native GPU particle fields now follow compacted applied collision pairs, actual
  source slot selections, and sparse overlaps recorded by the native resampling
  sweep. Additive source/removal ledgers, passive integer births/retirement and
  active-only field bounds run in retained device operations. Custom writers
  cannot change slot activity; undefined passive mergers/resampling, volume
  scaling and unbound ion events reject before upload, including restart.
- Particle-event qualification: 119 tests passed with one intentional Warp-CPU
  capture skip across particle fields, native exhaustion and prepared restart.
  Expanded field suite passed 28 tests on Warp CPU and real CUDA, including
  exact interval-remap values, nonzero collisions, separate scalar/vector
  inventories, passive int32/int64 births, failure suppression, inert capture,
  allocation/readback-free steps and byte-exact field/ledger restart. Targeted
  Ruff and mypy passed. Typed built-in contexts, maintained examples, broader
  qualification and the PR remain outstanding.
- Combined affected extension/prepared/exhaustion regression completed with
  285 passed and five skips; real CUDA particle-event/capture tests passed.

- Built-in process preparation now receives a typed resource context instead of
  the simulation owner. Native adapters use the common exact-type registry and
  return retained PreparedOperation records; cohort and thermodynamic composition
  preserve records, capabilities and resources. Lifecycle/transfer orchestration
  remains with the simulation owner, while physics preparation cannot call its
  step, snapshot, checkpoint or close methods. Native scientific admission and
  checkpoint configuration formats remain unchanged.
- Refactor qualification: 289 affected tests passed with one intentional
  Warp-CPU capture skip, including native/charged/atmospheric/cohort processes,
  particle fields, custom adapters, transfers and real CUDA capture/restart.
  Ruff and targeted mypy for all 11 changed execution modules pass. Published
  examples, full final qualification, performance evidence and the PR remain.
- The first maintained example now demonstrates two-box SciPy chemistry,
  time-aligned scheduled emissions via FunctionProcess, native dilution and a
  registered per-box injected-mass ledger. Every observation is checked against
  closed-form species evolution and volume-aware inventory. The complete CPU
  script passes with isolated Python from a temporary directory outside the
  checkout. Saved numerical data and inspected plots are published through the
  example registry; saved-data-only replot and documentation structure checks
  pass. Particle-tracer and resident/transfer examples remain outstanding.
