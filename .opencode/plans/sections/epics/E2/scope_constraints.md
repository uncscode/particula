## Scope and Constraints

### In Scope

- Container schema and ownership decisions for particle, gas, and environment
  state.
- CPU `EnvironmentData` and Warp `WarpEnvironmentData` implementations.
- Conversion helpers between CPU and GPU environment containers.
- Gas schema drift resolution for names, partitioning, vapor pressure, and
  round-trip semantics.
- Scalar-to-per-box migration helpers for existing GPU kernel APIs.
- Numerical studies for mass representation precision and condensation
  timestep stiffness.
- Documentation of CPU dynamics support boundaries for data containers.
- Foundation documentation and examples for downstream roadmap handoff.

### Out of Scope

- Full multi-box support for every CPU dynamics strategy.
- Complete replacement of legacy `ParticleRepresentation`, `GasSpecies`, or
  `Atmosphere` APIs.
- Production implementation of new stiff condensation integrators beyond the
  recommendation and foundation work described by E2-F7.
- Automatic hidden CPU/GPU synchronization in simulation loops.
- Broad performance optimization of all Warp kernels unrelated to environment
  migration and numerical foundation questions.

### Constraints

- Existing public APIs must remain compatible unless a child plan explicitly
  documents a deprecation path.
- Tests must run meaningfully on Warp CPU; CUDA-specific behavior remains
  optional where existing test conventions allow it.
- Shape validation should follow current `ParticleData` and `GasData` patterns:
  explicit `ValueError` messages with expected and actual shapes.
- Documentation must distinguish container capability from dynamics support.
