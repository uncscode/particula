# Implementation Tasks

## GPU Backend

- [ ] Define fixed-four count and stable scratch shape contracts in
  `particula/gpu/kernels/condensation.py`.
- [ ] Generalize buffer validation for work, accumulator, viscosity, and mean
  free path, with all checks completed before mutation.
- [ ] Extend `condensation_step_gpu()` using optional keyword-only scratch inputs
  without breaking current call forms.
- [ ] Clear reusable output/work buffers only after validation succeeds.
- [ ] Launch E4-F1 refresh and applicable environment preparation inside each
  of four equal substeps.
- [ ] Calculate, clamp, apply, and accumulate transfer each iteration.
- [ ] Return accumulated total transfer by caller-owned identity and preserve
  gas concentration unchanged.
- [ ] Confirm the all-scratch-supplied path performs no required allocation or
  hidden CPU/Warp synchronization.

## Tooling / Tests

- [ ] Add production scratch reuse and rejection tests to
  `particula/gpu/kernels/tests/condensation_test.py`.
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
