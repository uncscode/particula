## Implementation Strategy

### Architecture Overview

E2 extends particula's data/behavior split. Data containers hold arrays and
metadata; physics behavior remains in strategies, property functions, and Warp
kernels. The current reference patterns are `ParticleData`, `GasData`,
`WarpParticleData`, `WarpGasData`, and explicit conversion helpers in
`particula/gpu/conversion.py`.

### Key Data Ownership Rules

- The first dimension of batched state is `n_boxes`.
- Particle state owns per-particle/per-species masses, concentrations, charge,
  material density, and current per-box volume until E2-F1 decides otherwise.
- Gas state owns gas species names, molar masses, concentrations, and
  partitioning metadata.
- Environment state should own per-box thermodynamic state such as temperature,
  pressure, humidity, and saturation-related fields approved by E2-F1.
- Vapor pressure semantics must be explicit: either passed as computed kernel
  input, represented as gas/environment state, or intentionally excluded from
  round trips with documented restoration requirements.

### Reusable Patterns

- CPU containers use dataclasses, `np.asarray(..., dtype=...)`, exact shape
  validation, `ValueError` with expected/got details, and deep `copy()` methods.
- Builders may provide single-box convenience by inserting batch dimensions.
- Warp structs use `wp.float64` arrays and are imported/exported only when Warp
  is available.
- Conversion helpers are explicit and tested on CPU Warp devices, with CUDA as
  optional availability.

### Testing Requirements

1. Test coverage thresholds must NEVER be lowered
2. Each phase must include self-contained tests
3. Tests are committed in the same PR as the implementation
4. Test files use `*_test.py` suffix in module-level `tests/` directories
5. Minimum 80% coverage (configured in `pyproject.toml`)

### Technical Approach

- Start with schema and docs decisions in E2-F1 to minimize API churn.
- Implement `EnvironmentData` using `ParticleData`/`GasData` validation style.
- Add `WarpEnvironmentData` and conversions beside existing GPU types and
  conversion helpers.
- Reconcile gas schema drift with tests that lock in accepted behavior before
  downstream code depends on it.
- Add normalization helpers that accept scalar temperature/pressure or per-box
  environment state while preserving existing kernel signatures.
- Treat E2-F6 and E2-F7 as evidence-producing tracks that can recommend future
  implementation work without overextending E2 scope.
