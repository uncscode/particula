# E2-F6 Infrastructure Reuse

## Existing Data Containers

- `particula/particles/particle_data.py` provides the CPU reference container.
  `ParticleData.masses` is `NDArray[np.float64]` with shape
  `(n_boxes, n_particles, n_species)`.
- `particula/particles/particle_data_builder.py` normalizes mass inputs with
  `np.asarray(..., dtype=np.float64)` and should be cited as current behavior.
- `particula/gpu/warp_types.py` defines `WarpParticleData.masses` as
  `wp.array3d(dtype=wp.float64)`.
- `particula/gpu/conversion.py` transfers CPU particle data to Warp with
  explicit `wp.float64` arrays.

## Existing Physics and Numerical References

- `particula/dynamics/condensation/mass_transfer.py` contains CPU reference
  mass-transfer and conservation-limiting logic. Use it as the conservation
  reference where possible.
- `particula/gpu/kernels/condensation.py` is the current GPU path to profile and
  compare. It uses `wp.float64` kernels and clamps negative particle masses.
- `particula/particles/distribution_strategies/particle_resolved_speciated_mass.py`
  shows legacy particle-resolved mass behavior and add-mass semantics.

## Existing Tests and Benchmarks

- `particula/dynamics/condensation/tests/staggered_mass_conservation_test.py`
  includes small-particle fixtures and conservation assertions.
- `particula/dynamics/condensation/tests/staggered_stability_test.py` provides
  finite/nonnegative/stability patterns.
- `particula/gpu/tests/benchmark_test.py` provides GPU scaling cases and output
  conventions.
- `particula/gpu/kernels/tests/condensation_test.py` provides CPU/GPU parity and
  clamping test patterns.

## Documentation Sources

- `docs/Features/Roadmap/data-oriented-gpu.md` is the roadmap source for T6.
- `docs/Features/particle-data-migration.md` describes data-container migration
  context and should cross-link the final recommendation if updated.
