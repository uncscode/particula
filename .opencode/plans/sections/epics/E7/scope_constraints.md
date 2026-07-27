# Scope and Constraints

## In Scope

- Typed backend-selection and execution-context API with a capability matrix.
- Backend-selected condensation and Brownian coagulation adapters.
- GPU-resident session ownership for particle, gas, environment, reusable
  sidecars, diagnostics, and stochastic state.
- Explicit setup, timestep, checkpoint, restart, and finalization operations.
- Deterministic full-process scheduling for supported condensation,
  coagulation, dilution, wall loss, and nucleation paths.
- Correct environment, vapor-pressure, saturation, and gas update ordering.
- Explicit CPU fallback boundaries, capability errors, public exports, and API
  stability rules.
- Fixed-shape multi-box transport maps, gas and particle transport, mixing,
  advection, and volume/expansion updates.
- Persistent per-box RNG streams and restart semantics.
- Diagnostics, full-loop regressions, support documentation, and a complete
  multi-timestep example.

## Out of Scope

- Silent fallback, hidden transfer, unsupported physics expansion, or kernel
  physics rewrites.
- GPU staggered condensation, dynamic resizing/compaction, multi-GPU,
  distributed execution, or full CFD coupling.
- Epic H graph capture, profiling, benchmarks, and optimization.
- Epic I autodiff and inverse/optimization workflows.

## Constraints

- Issue #1451 and Epic G in
  `docs/Features/Roadmap/data-oriented-gpu.md:1461-1593` are scope authority.
- Python 3.12+, NumPy, and Warp; single-device execution only.
- Preserve fixed shapes, array and container identities, deliberate exports,
  explicit synchronization, and caller-visible ownership.
- Warp CPU is the routine parity backend; CUDA validation is optional and must
  skip cleanly when unavailable.
- CPU implementations remain independent references. Conservation and parity
  tolerances must be explicit; stochastic validation is statistical or
  stream-contract based, not exact CPU/CUDA trajectory matching.
