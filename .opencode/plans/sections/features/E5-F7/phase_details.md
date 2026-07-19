# Phase Details

- [x] **E5-F7-P1:** Publish independent deterministic pair-parity matrix and fixtures
  - Issue: #1362 | Size: S | Status: Shipped
  - Goal: Parameterize every shipped single and approved additive row against
    independent fp64 pair/property equations and explicit majorant checks.
  - Files: `particula/gpu/kernels/tests/coagulation_validation_test.py` and
    `particula/gpu/kernels/tests/_coagulation_validation_support.py`
  - Tests: Literal fixed-mask and deferred-mask configuration checks; independent
    fp64 fixture/oracle validation; lazy Warp-CPU property, pair symmetry,
    finite/non-negative, summed-rate, and selector-majorant observations;
    active-index and exact-zero boundaries.
  - Result: Validation-only commit; no production, public API, or user-doc change.

- [x] **E5-F7-P2:** Validate cross-mechanism conservation, ownership, and edge cases
  - Issue: #1363 | Size: S | Status: Shipped
  - Goal: Run the executable public-entry matrix through one-/two-box and
    one-/two-species materializations while proving inventory, applicable charge,
    and caller-ownership invariants.
  - Files: `particula/gpu/kernels/tests/coagulation_validation_test.py` and the
    shared private `particula/gpu/kernels/tests/_coagulation_public_step_support.py`
    after its P3 extraction.
  - Tests: Every executable mask; per-box/per-species inventory and charge;
    legal pair prefixes, donor/recipient bookkeeping, inactive sentinels, sparse
    and two-active boundaries, zero-rate no-ops, capacity rejection, pair/count
    ownership, RNG lifecycle, turbulent scalar/device inputs, and selected
    preflight atomicity. Warp CPU runs when installed; CUDA is optional.
  - Result: Validation-only commit; no production, public API, or user-doc change.

- [x] **E5-F7-P3:** Publish bounded stochastic and device validation matrix
  - Issue: #1364 | Size: S | Status: Shipped
  - Goal: Establish documented aggregate stochastic bounds for each executable
    row and execute the same correctness cases on required Warp CPU and optional
    CUDA.
  - Files: `particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py`,
    `_coagulation_validation_support.py`, and the new private
    `_coagulation_public_step_support.py`; the existing CUDA helper is reused
    without modification.
  - Tests: Independent host-only bounded-case/oracle and override regression
    coverage plus 100 fresh seeded public runs for every executable mask/device.
    Each device observation applies P2 invariants and accepts only aggregate
    counts within `3 * sqrt(expected_mean)`; Warp CPU is required when installed
    and CUDA skips cleanly when unavailable.
  - Result: Validation-only commit; no production, public API, export, shared
    CUDA-helper, or user-documentation change.

- [x] **E5-F7-P4:** Update development documentation
  - Issue: #1365 | Size: XS | Status: Shipped
  - Goal: Published the support/evidence matrix, limitations, and focused
    reproduction commands for E5-F9 and future release reviewers.
  - Files: `.opencode/guides/testing_guide.md`, relevant
    `docs/Features/Roadmap/` GPU roadmap document, and E5-F7 plan sections
  - Result: Documentation and plan-state metadata were synchronized; Markdown
    links, commands, markers, mechanism rows, required Warp CPU, and optional
    CUDA wording were validated.
