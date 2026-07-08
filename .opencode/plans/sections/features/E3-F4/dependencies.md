# Dependencies

## Internal Dependencies

- Parent epic: `E3`.
- Required predecessor: `E3-F1`, because the coagulation quick-start should
  not demonstrate repeated persisted `rng_states` behavior until seed-once RNG
  initialization semantics are settled.
- Related sibling features:
  - `E3-F2` may influence future coagulation documentation if mixed-scale
    sampling behavior changes.
  - `E3-F3` provides benchmark and scaling evidence that can inform
    troubleshooting/performance notes.

## Code Dependencies

- `particula/gpu/__init__.py`
- `particula/gpu/kernels/__init__.py`
- `particula/gpu/kernels/condensation.py`
- `particula/gpu/kernels/coagulation.py`
- `particula/gpu/kernels/environment.py`
- `particula/gpu/conversion.py`
- `particula/gpu/warp_types.py`

## Test Dependencies

- `particula/gpu/tests/warp_types_test.py`
- `particula/gpu/tests/conversion_test.py`
- `particula/gpu/tests/data_containers_example_test.py`
- `particula/gpu/kernels/tests/condensation_test.py`
- `particula/gpu/kernels/tests/coagulation_test.py`

## Documentation Dependencies

- `docs/Examples/data_containers_and_gpu_foundations.py`
- `docs/Features/data-containers-and-gpu-foundations.md`
- `docs/Features/Roadmap/data-oriented-gpu.md`

## External Dependencies

- `warp-lang` for Warp-backed execution.
- CUDA-capable device is optional. The quick-start and tests must support
  `device="cpu"` and skip CUDA-only behavior cleanly.

## Phase Ordering Notes

- P1 must settle the supported import surface before broader regression tests or
  user-facing examples are expanded, otherwise later phases risk documenting an
  unstable public path.
- P2 follows P1 because import/export regression coverage should lock the chosen
  public surface before the quick-start depends on it.
- P3 follows P1/P2 so the runnable example uses the same import path, transfer
  boundary, and RNG expectations already protected by tests.
- P4 ships last after the example and API surface are stable, because
  troubleshooting text and release-facing validation guidance should reference
  the final documented quick-start rather than provisional behavior.
