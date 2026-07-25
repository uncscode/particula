# Architecture Outline

## Particle Package

`particula/particles/` contains particle-data representations, distribution
strategies, and focused particle-domain helpers.

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

### particula/dynamics/nucleation/

CPU-only P4 construction boundary for nucleation potential-rate
parameterizations. P2 source-demand planning and P3 transaction helpers remain
concrete-module-only.

**Key Components:**
- `nucleation_strategies.py` - Immutable scalar configuration records and
  activation/kinetic potential-rate algorithms. The P4 records, builders, and
  factory are deliberately exported through `particula.dynamics.nucleation`
  and `particula.dynamics`; strategies still return rates only and own no
  source admission, state mutation, runnable, or GPU integration.
- `nucleation_configuration.py`, `nucleation_builders.py`, and
  `nucleation_factories.py` - Strict P4 source-selection metadata and fresh,
  unit-normalizing construction APIs. They do not import or expose P2/P3
  particle-source records or transaction helpers.
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
