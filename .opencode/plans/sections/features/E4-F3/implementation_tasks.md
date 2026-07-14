# Implementation Tasks

## GPU Backend

- [x] Define `CondensationScratchBuffers` and stable scratch shape contracts in
  `particula/gpu/kernels/condensation.py` (issue #1292).
- [x] Validate work, total-transfer, viscosity, and mean-free-path sidecars as
  one pre-mutation metadata gate (issue #1292).
- [x] Extend `condensation_step_gpu()` with keyword-only `scratch_buffers`
  without breaking existing positional calls (issue #1292).
- [x] Resolve partial sidecars only after validation; preserve legacy
  `mass_transfer` identity and reject transfer-storage overlap (issue #1292).
- [ ] Launch E4-F1 refresh and applicable environment preparation inside each
  of four equal substeps.
- [ ] Calculate, clamp, apply, and accumulate transfer each iteration.
- [ ] Return accumulated total transfer by caller-owned identity and preserve
  gas concentration unchanged.
- [ ] Confirm the all-scratch-supplied path performs no required allocation or
  hidden CPU/Warp synchronization.

## Tooling / Tests

- [x] Add co-located production scratch reuse, partial-sidecar, overlap, and
  pre-mutation rejection coverage in
  `particula/gpu/kernels/tests/_condensation_test_support.py`, exported by
  `particula/gpu/kernels/tests/condensation_test.py` (issue #1292).
- [ ] Promote fixed-four candidate cases through
  `condensation_stiffness_test.py` against the production entry point.
- [ ] Assert exactly four equal iterations and per-substep E4-F1 refresh.
- [ ] Assert deterministic, finite, nonnegative, total-transfer, and unchanged
  gas semantics on repeated calls.
- [ ] Preserve the recorded nanometer, accumulation-mode, and droplet-like
  timestep-grid bound of `5e-2` without presenting it as universal tolerance.
- [ ] Run Warp CPU coverage and optional CUDA coverage with clean skips.

## Documentation

- [ ] Update the stiffness study from test-local recommendation to shipped
  production contract, retaining evidence and limitations.
- [ ] Update the data-oriented GPU roadmap and user-facing scratch guidance.
