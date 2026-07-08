# Testing Strategy

Testing is co-located with each implementation phase. Do not create a separate
testing-only phase.

## Import and Export Tests

- Add tests for the chosen direct-kernel import path.
- If top-level exports are added, assert both step functions are present in
  `particula.gpu.__all__`.
- If `.kernels` remains the only public kernel module, assert the documented
  path imports successfully and remains stable.
- Keep tests independent of CUDA availability.

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
pytest particula/gpu/tests/kernel_exports_test.py -q
pytest particula/gpu/tests/data_containers_example_test.py -q
pytest particula/gpu/kernels/tests/condensation_test.py -q
pytest particula/gpu/kernels/tests/coagulation_test.py -q
```

Run full GPU-focused tests before shipping if code exports change:

```bash
pytest particula/gpu/tests particula/gpu/kernels/tests -q
```
