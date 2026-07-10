# Testing Strategy

Testing is co-located with each implementation phase. Do not create a separate
testing-only phase.

## Import and Export Tests

- Phase `E3-F4-P1` shipped focused regression coverage in
  `particula/gpu/tests/kernel_exports_test.py`.
- Phase `E3-F4-P2` made that file the single canonical home for package-surface
  checks and removed duplicate export assertions from
  `particula/gpu/kernels/tests/coagulation_test.py`.
- Assert `particula.gpu.kernels` imports
  `condensation_step_gpu` and `coagulation_step_gpu` via parametrized checks
  without launching kernels.
- Assert top-level `particula.gpu` does not expose those step functions and does
  not list them in `__all__`.
- Assert `particula.gpu.kernels.__all__` contains exactly the two supported
  step functions.
- Assert representative internal helpers
  (`apply_coagulation_kernel`, `apply_mass_transfer_kernel`,
  `condensation_mass_transfer_kernel`, and
  `initialize_coagulation_rng_states`) are absent from the package surface while
  a representative helper still resolves from its concrete module.
- Keep the positive `particula.gpu.kernels` checks independent of CUDA
  availability and guarded with `pytest.importorskip("warp")`.

## Quick-Start Smoke Tests

- Follow the pattern in `particula/gpu/tests/data_containers_example_test.py`.
- Verify the example has a clean no-Warp path.
- Verify the example runs with Warp on `device="cpu"` when `WARP_AVAILABLE` is
  true.
- Use optional CUDA checks only when CUDA is available; otherwise skip without
  failing CI.

## Transfer Boundary Assertions

- Tests should ensure example code calls explicit `to_warp_*` and
  `from_warp_*` helpers rather than relying on hidden transfer behavior.
- Kernel tests should continue to validate that direct environment inputs and
  explicit `environment=` are not mixed.
- Device mismatch troubleshooting should be backed by existing validation tests
  or new focused tests if the docs introduce new behavior.

## Suggested Commands

```bash
pytest particula/gpu/tests/kernel_exports_test.py -q -Werror
pytest particula/gpu/tests/data_containers_example_test.py -q
pytest particula/gpu/kernels/tests/condensation_test.py -q
pytest particula/gpu/kernels/tests/coagulation_test.py -q
```

Run full GPU-focused tests before shipping if code exports change:

```bash
pytest particula/gpu/tests particula/gpu/kernels/tests -q
```
