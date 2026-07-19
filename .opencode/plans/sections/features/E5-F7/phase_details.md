# Phase Details

- [x] **E5-F7-P1:** Publish independent deterministic pair-parity matrix and fixtures
  - Issue: #1362 | Size: S | Status: Implemented
  - Goal: Parameterize every shipped single and approved additive row against
    independent fp64 pair/property equations and explicit majorant checks.
  - Files: `particula/gpu/kernels/tests/coagulation_validation_test.py` and
    `particula/gpu/kernels/tests/_coagulation_validation_support.py`
  - Tests: Literal fixed-mask and deferred-mask configuration checks; independent
    fp64 fixture/oracle validation; lazy Warp-CPU property, pair symmetry,
    finite/non-negative, summed-rate, and selector-majorant observations;
    active-index and exact-zero boundaries.
  - Result: Validation-only commit; no production, public API, or user-doc change.

- [ ] **E5-F7-P2:** Validate cross-mechanism conservation, ownership, and edge cases
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Run the executable matrix through one-box/multi-box and active/inactive
    fixtures while proving tight mass/charge and caller-ownership invariants.
  - Files: `particula/gpu/kernels/tests/coagulation_validation_test.py`,
    `particula/gpu/kernels/tests/coagulation_test.py` where legacy coverage is
    extended
  - Tests: Per-box/per-species mass, separate charge, inactive slots, zero/one/
    two/many active particles, capacity, buffer identity, RNG reuse/reset, and
    preflight atomicity.

- [ ] **E5-F7-P3:** Publish bounded stochastic and device validation matrix
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Establish documented aggregate stochastic bounds for each executable
    row and execute the same correctness cases on required Warp CPU and optional
    CUDA.
  - Files: `particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py`,
    `_coagulation_validation_support.py`, and `particula/gpu/tests/cuda_availability.py`
    only if shared device support needs a compatible extension
  - Tests: Repeated fresh seeded runs using the neutral support table shared with
    `coagulation_validation_test.py`, expected collision aggregates, declared
    sigma/confidence windows, deterministic invariants on every run, Warp CPU
    enforcement, and clean CUDA skips/parity when available.

- [ ] **E5-F7-P4:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish the support/evidence matrix, limitations, and focused
    reproduction commands for E5-F9 and future release reviewers.
  - Files: `.opencode/guides/testing_guide.md`, relevant
    `docs/Features/Roadmap/` GPU roadmap document, and E5-F7 plan sections
  - Tests: Markdown links, command references, marker selection, mechanism-row
    names, required Warp CPU wording, and optional CUDA wording.
