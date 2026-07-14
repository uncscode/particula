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
- [x] Launch E4-F1 refresh and applicable environment preparation inside each
  of four equal substeps (issue #1293).
- [x] Calculate, clamp, apply, and accumulate transfer each iteration (issue
  #1293).
- [x] Return accumulated total transfer by caller-owned identity and preserve
  gas concentration unchanged (issue #1293).
- [x] Confirm recorded stiffness trials reuse one complete caller-owned scratch
  sidecar sequentially across fresh-input executions (issue #1294).

## Tooling / Tests

- [x] Add co-located production scratch reuse, partial-sidecar, overlap, and
  pre-mutation rejection coverage in
  `particula/gpu/kernels/tests/_condensation_test_support.py`, exported by
  `particula/gpu/kernels/tests/condensation_test.py` (issue #1292).
- [x] Promote fixed-four recorded cases through
  `condensation_stiffness_test.py` against the production entry point, with
  discovery restricted to production-prefixed tests (issue #1294).
- [x] Assert exactly four equal iterations and per-substep E4-F1 refresh
  (issue #1293).
- [x] Assert deterministic, finite, nonnegative, total-transfer, and unchanged
  gas semantics on repeated calls (issue #1293).
- [x] Preserve the recorded nanometer, accumulation-mode, and droplet-like
  timestep-grid bound of `5e-2` without presenting it as universal tolerance
  (issue #1294).
- [x] Add required Warp CPU coverage and one optional CUDA slice with clean
  skips (issue #1294).

## Documentation

- [x] Update the stiffness study from test-local recommendation to shipped
  production contract, retaining evidence and limitations (issue #1295).
- [x] Update the data-oriented GPU roadmap and user-facing scratch guidance,
  including the bounded low-level Warp note (issue #1295).
