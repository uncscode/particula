# Infrastructure Reuse

- `condensation_step_gpu()` in `particula/gpu/kernels/condensation.py`
  (currently line 794) remains the direct public operation and owns
  orchestration.
- `condensation_mass_transfer_kernel` in `condensation.py` (currently line 326)
  computes one substep from current particle and thermodynamic state; launch it
  four times.
- `apply_mass_transfer_kernel` in `condensation.py` (currently line 582)
  performs the in-place nonnegative update after each transfer calculation.
- `_validate_mass_transfer_buffer` in `condensation.py` (currently line 672)
  supplies the shape/device and fail-before-mutation pattern for all new scratch
  validators.
- E4-F1's fixed-shape on-device vapor-pressure refresh must be called inside the
  loop, before each transfer calculation.
- `CondensationCandidateScratch` in
  `particula/gpu/kernels/tests/_condensation_test_support.py` (currently line
  585) and the fixed-four prototype define stable work/accumulator ownership
  and total-transfer semantics.
- Candidate regressions at `_condensation_test_support.py:2839-3055` provide
  deterministic, finite, nonnegative, reuse, parity, and stiffness evidence.
- Production buffer tests at `_condensation_test_support.py:3270-3438` define
  caller identity, wrong-shape/device, and pre-mutation failure conventions.
- Fixed fp64 schemas in `particula/gpu/warp_types.py:24-169` remain unchanged;
  scratch buffers are operation-owned sidecars rather than container fields.
