# Architecture Outline

## Particle Package

`particula/particles/` contains particle-data representations, distribution
strategies, and focused particle-domain helpers.

### particula/particles/

**Key Components:**
- `particle_data.py` - Fixed-shape CPU particle-data container and conversion
  helpers
- `slot_management.py` - CPU-only fixed-slot classification, discovery, and
  direct-import activation; exports only `get_slot_diagnostics` through
  `particula.particles`. Activation preserves fixed capacity and excludes
  `ParticleData` API changes, GPU support, and a top-level particles export
- `distribution_strategies/` - Particle distribution representations and
  update strategies
- `properties/` - Particle property calculations
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
- `wall_loss.py` - Concrete fixed-slot neutral/charged GPU wall-loss boundary;
  owns immutable host configuration, frozen preflight, bounded fixed-slot
  removal, and the external caller-owned per-box RNG sidecar lifecycle. Charged
  mode composes private image-charge and field-drift helpers from
  `particula.gpu.dynamics.wall_loss_funcs` only for nonzero-charge slots;
  zero-charge slots retain the neutral path. The sidecar is not added to Warp
  particle schemas or package exports, and sequential per-box ownership
  advances it only for eligible slots.
- `slot_management.py` - Concrete-module-only, read-only direct-Warp
  fixed-slot diagnostics. It classifies particle mass, concentration, and
  charge into caller-owned `int32` diagnostic sidecars without accessing
  density or volume; it is deliberately not exported by
  `particula.gpu.kernels`.
- `environment.py` - Shared private normalization and validation for kernel
  environment inputs
- `tests/` - Test coverage
