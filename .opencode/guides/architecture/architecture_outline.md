# Architecture Outline

## Execution Capability Vocabulary

`particula.execution` is a deliberately dependency-neutral metadata and private
selection boundary. It is standard-library-only and is not exported through
top-level `particula`. Its P2 selection layer validates declared capability
support, normalizes the CPU spelling, and returns one context-local adapter by
the exact `(Process, Backend)` key. It does not import Warp or `particula.gpu`,
resolve or probe backends, transfer data, invoke adapters, or provide fallback.

### particula/execution.py

**Key Components:**
- `Backend`, `Device`, `Process`, and `Capability` - Immutable typed metadata;
  `Device.native` is an opaque native identifier.
- `CapabilityRequirements` and `CapabilityDeclaration` - Immutable exact
  capability-support declarations.
- `CapabilityMatrix` - Pure, immutable exact-match lookup. Nonempty
  requirements must match a complete declaration; an empty requirement is
  supported only when its device/process base has a declaration.
- `ExecutionRequest` and `ExecutionContext` - Direct-import selection values
  and context-local coordinator. Requests must pair matching backends/devices;
  CPU selection accepts only `Device(Backend.CPU, "cpu")`, while Warp native
  identifiers remain opaque.
- `_AdapterRegistry` and `_ExecutionAdapter` - Private, per-context exact-key
  registration seam. Adapters are selected by identity only after matrix
  validation; registration and execution are deliberately not public APIs.

## Particle Package

`particula/particles/` contains particle-data representations, distribution
strategies, and focused particle-domain helpers.

## Direct GPU Nucleation

`particula/gpu/kernels/nucleation.py` implements package-exported direct-Warp
`nucleation_step_gpu`: P1 preflight, P2 admission, P3 fixed-slot staging, P4
resampling-first/scaling-fallback, and fused P5 selected-slot/gas transfer.
Only the step is exported; configuration, records, sidecars, and helpers remain
concrete-only. It has no hidden transfer, CPU fallback, resize/compaction, GPU
Runnable, or E6-F9 integration.

### particula/particles/

**Key Components:**
- `exhaustion.py` - Concrete, deliberately unexported CPU P1 read-only
  fixed-shape capacity exhaustion planning boundary, P2 validated resampling
  apply commit, and P4 direct all-box-preflighted representative-volume scaling
  commit with caller-owned sidecars and float64 weighted
  inventory accounting. P1 validates every box before resolution, applies
  resampling-first deferred-policy selection, and returns immutable plans
  without mutating state. P2 and P4 each own their separate CPU commit
  boundaries; the module owns no GPU work or container schema.
- `distribution_strategies/` - Particle distribution implementations
- `particle_data.py` - Fixed-shape CPU particle-data container and conversion
  helpers
- `slot_management.py` - CPU-only fixed-slot classification, discovery, and
  direct-import activation; exports only `get_slot_diagnostics` through
  `particula.particles`. Activation preserves fixed capacity and excludes
  `ParticleData` API changes, GPU support, and a top-level particles export
- `properties/` - Particle property calculations
- `tests/` - Test coverage

## Dynamics Package

`particula/dynamics/` contains physics-domain calculations and narrowly scoped
implementation boundaries.

### particula/dynamics/

**Key Components:**
- `particle_process.py` - Public CPU-only, single-box `Nucleation` runnable
  boundary and immutable `NucleationCommitConfig`. `Nucleation` adapts legacy
  `Aerosol` particle and partitioning-gas backing data by identity, accepts P4
  source configuration plus `EnvironmentData`, and runs equal, gas-coupled
  substeps. Each attempted substep has P3 transaction atomicity; prior
  successful substeps remain applied if a later substep fails. This boundary
  introduces no GPU or broader runnable orchestration.

### particula/dynamics/nucleation/

CPU-only P4 construction boundary for nucleation potential-rate
parameterizations, consumed by the supported P5 `Nucleation` process boundary
in `particula.dynamics`. P2 source-demand planning and P3 transaction helpers
remain concrete-module-only.

**Key Components:**
- `nucleation_strategies.py` - Immutable scalar configuration records and
  activation/kinetic potential-rate algorithms. The P4 records, builders, and
  factory are deliberately exported through `particula.dynamics.nucleation`
  and `particula.dynamics`; strategies still return rates only and own no
  source admission, state mutation, runnable, or GPU integration. The P5
  runnable composes them without changing their ownership.
- `nucleation_configuration.py`, `nucleation_builders.py`, and
  `nucleation_factories.py` - `NucleationSourceConfig`, its builder, and the
  activation/kinetic builders and factory form the strict, fresh,
  unit-normalizing P4 construction API. They do not import or expose P2/P3
  particle-source records or transaction helpers.
- `particle_source.py` - Concrete-module-only P2 source-demand planning and P3
  atomic particle-source transaction records and helpers. It is intentionally
  absent from both `particula.dynamics.nucleation` and `particula.dynamics`
  exports.
- `tests/` - Test coverage

## GPU Package

`particula/gpu/` contains Warp-backed data containers, explicit CPU↔GPU
transfer helpers, device-side physics helpers, and kernel entry points.

### particula/gpu/

**Key Components:**
- `__init__.py` - Public GPU exports
- `conversion.py` - Explicit CPU↔GPU transfer helpers only
- `warp_types.py` - Warp container schemas only
- `dynamics/` - GPU physics helper functions
- `properties/` - GPU property helper functions
- `kernels/` - GPU kernel entry points and private kernel support helpers
- `tests/` - Test coverage

### particula/gpu/kernels/

GPU kernel entry points own launch-time orchestration and may depend on shared
private helpers for cross-kernel setup.

**Key Components:**
- `condensation.py` - Condensation GPU entry points and kernels
- `coagulation.py` - Coagulation GPU entry points and kernels
- `dilution.py` - Concrete P1 GPU dilution input boundary; validation scans may
  allocate or launch, but rejected calls have no update-kernel launch or caller
  mutation
- `exhaustion.py` - Direct Warp fixed-capacity equal-weight resampling boundary;
  `resampling_step_gpu` remains the only exhaustion package export. Resampling
  consumes explicit per-box release counts, uses caller-owned planning and
  diagnostic buffers, and atomically commits only after all boxes pass
  diagnostics. The concrete-only P4 representative-volume scaling helper uses
  caller-owned sidecars and a separate all-box-preflighted scaling commit; it
  adds no policy, transfer, resizing, or runnable behavior. Only
  `resampling_step_gpu` is exported; `ResamplingBuffers`, P4 sidecars, status
  codes, and kernels remain concrete-module-only. Neither boundary provides a
  runnable, policy resolution, CPU fallback or transfer, or resizing.
- `nucleation.py` - Concrete E6-F8 implementation of package-exported
  `nucleation_step_gpu` for fixed-capacity GPU nucleation. P1 preflights; P2
  calculates and inventory-admits `E_pot=J*dt`, with survival already in `J`;
  P3 stages demand and fixed slots; P4 uses resampling before
  representative-volume-scaling fallback; and fused P5 commits selected slots
  and matching gas transfer. P3 retains counts beyond free capacity and emits
  ascending free-slot prefixes with `-1` tails. Only the step is exported;
  configuration, records, sidecars, and helpers remain concrete-only. Public
  rejection before P4 primitive entry preserves particle and gas state, though
  P2--P4 may have written documented sidecars before a later rejection. No
  rollback is promised after an E6-F6 primitive entry or P5 writer launch. The
  step provides no hidden transfer, CPU fallback, resize/compaction, GPU
  `Runnable`, or E6-F9 integration.
- `wall_loss.py` - Concrete fixed-slot neutral/charged GPU wall-loss boundary;
  owns immutable host configuration, frozen preflight, bounded fixed-slot
  removal, and the external caller-owned per-box RNG sidecar lifecycle. Charged
  mode composes private image-charge and field-drift helpers from
  `particula.gpu.dynamics.wall_loss_funcs` only for nonzero-charge slots;
  zero-charge slots retain the neutral path. The sidecar is not added to Warp
  particle schemas or package exports, and sequential per-box ownership
  advances it only for eligible slots.
- `slot_management.py` - Concrete-only P3 read-only direct-Warp diagnostics
  classify particle mass, concentration, and charge into caller-owned `int32`
  sidecars without accessing density or volume. Package-exported P4
  `activate_slots_gpu` maps selected request prefixes to ascending
  fixed-capacity free slots. It reads and writes only caller-owned mass,
  concentration, and charge storage; its activation and diagnostics sidecars
  are caller-owned device `int32` arrays. P4 completes preflight before its
  writer launches, makes no hidden transfers, and does not promise rollback
  after a launched writer.
- `environment.py` - Shared private normalization and validation for kernel
  environment inputs
- `tests/` - Test coverage
