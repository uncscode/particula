# Infrastructure Reuse

## Existing code to leverage

- `particula/gpu/warp_types.py`: defines `WarpParticleData` and `WarpGasData`
  with `@wp.struct` and Warp array field annotations.
- `particula/gpu/conversion.py`: contains `_ensure_warp_available`,
  `_validate_device`, `to_warp_particle_data`, `from_warp_particle_data`,
  `to_warp_gas_data`, and `from_warp_gas_data` patterns.
- `particula/gpu/__init__.py`: centralizes optional Warp availability checks,
  lazy public exports, and `__all__` updates.
- `particula/gpu/tests/warp_types_test.py`: provides optional Warp import
  gating and struct test organization.
- `particula/gpu/tests/conversion_test.py`: provides CPU backend conversion,
  round-trip, error, copy, and sync test patterns.
- `particula/gpu/tests/cuda_availability.py`: provides `warp_devices(wp)` for
  CPU plus optional CUDA parametrization.
- CPU containers such as `particula/gas/gas_data.py` and
  `particula/particles/particle_data.py`: establish `np.float64` coercion,
  first-dimension `n_boxes`, and deep-copy conventions.

## Reuse rules

- Use the existing Warp optional dependency error message path instead of
  adding a separate import guard.
- Use `wp.array(..., dtype=wp.float64, device=device)` for `copy=True` and
  `wp.from_numpy(..., dtype=wp.float64, device=device)` for `copy=False`.
- Use `.numpy()` only inside the explicit `from_warp_environment_data` helper.
- Keep documentation and tests aligned with established `*_test.py` naming and
  `pytest.importorskip("warp")` behavior.
