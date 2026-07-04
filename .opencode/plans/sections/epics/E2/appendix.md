## Appendix

### Relevant Code Areas

- `particula/particles/particle_data.py` -- CPU particle container schema,
  validation, derived quantities, and copy behavior.
- `particula/gas/gas_data.py` -- CPU gas container schema, validation,
  partitioning casting, and copy behavior.
- `particula/gpu/warp_types.py` -- Warp particle and gas structs; target home
  for `WarpEnvironmentData`.
- `particula/gpu/conversion.py` -- CPU/GPU transfer helpers and gas round-trip
  behavior.
- `particula/gpu/kernels/condensation.py` -- scalar temperature/pressure
  condensation API and explicit mass transfer step.
- `particula/gpu/kernels/coagulation.py` -- scalar temperature/pressure
  coagulation API and per-box volume normalization precedent.
- `particula/dynamics/condensation/condensation_strategies.py` -- current CPU
  data-container boundary and single-box guard.

### Relevant Documentation

- `docs/Features/Roadmap/data-oriented-gpu.md` -- roadmap source for Epic A
  scope and known gaps.
- `docs/Features/particle-data-migration.md` -- existing data/behavior split
  and migration guide.
- `docs/Features/Roadmap/warp-autodiff-limitations.md` -- related GPU/Warp
  limitations context.

### Test Patterns

- Container tests live in module-level `tests/` directories and use the
  `*_test.py` suffix.
- GPU tests should remain meaningful on Warp CPU and use availability helpers
  for CUDA-specific coverage.
- Shape mismatch tests should assert clear error paths rather than only happy
  paths.
