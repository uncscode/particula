# Infrastructure Reuse

## Existing Containers and Builders

- Reuse `ParticleData` and related tests in
  `particula/particles/particle_data.py` and
  `particula/particles/tests/particle_data_test.py` as the current particle
  schema source.
- Reuse `GasData` and related tests in `particula/gas/gas_data.py` and
  `particula/gas/tests/gas_data_test.py` as the current gas schema source.
- Reuse builder tests to document user-facing broadcasting and unit-conversion
  behavior where that behavior affects shape conventions.

## Existing GPU Types and Conversion Helpers

- Reuse `WarpParticleData` and `WarpGasData` in
  `particula/gpu/warp_types.py` as the current GPU schema source.
- Reuse `particula/gpu/conversion.py` to document explicit CPU/GPU transfer
  behavior, including lossy gas round trips for `name` and `vapor_pressure`.
- Reuse `particula/gpu/tests/warp_types_test.py` and
  `particula/gpu/tests/conversion_test.py` as checkable examples for shape and
  dtype expectations.

## Existing Documentation

- Extend or link from `docs/Features/Roadmap/data-oriented-gpu.md`, which
  already identifies schema drift and the need for environment containers.
- Reuse `docs/Features/particle-data-migration.md` for data/behavior split
  context and user-facing migration examples.
- Follow repository documentation conventions for feature docs and any decision
  record location selected by reviewers.

## Validation Patterns

- Preserve leading `n_boxes` as the first dimension for mutable batched state.
- Preserve `float64` as the current data-container dtype norm unless E2-F6
  later changes precision policy.
- Use existing validation patterns in container constructors and builders when
  documenting allowed shapes.
