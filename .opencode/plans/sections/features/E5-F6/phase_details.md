# Phase Details

- [x] **E5-F6-P1:** Define additive combination matrix and preflight validation with unit tests
  - Issue: #1357 | Size: S | Status: Implemented
  - Delivered: A private immutable recognition table accepts four singleton,
    six two-way, and one four-way mask. Three-way masks reject before particle
    access. Recognition is separate from executable support: valid deferred
    masks complete enabled-term read-only preflight then raise the stable
    deferred-execution error before downstream work or mutation.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: matrix/order and structural rejection, per-term validation, deferred
    mask launch/helper bypass, and snapshots proving particles, outputs, and
    persistent RNG remain unchanged.

- [x] **E5-F6-P2:** Implement safe summed majorants and additive pair dispatch with unit tests
  - Issue: #1358 | Size: S | Status: Implemented
  - Delivered: Private checked fp64 addition and all-enabled-bit pair-rate and
    majorant dispatch for masks `3`, `5`, `6`, `9`, `10`, `12`, and `15`.
    Invalid/nonpositive/overflowed aggregates fail closed to zero. A private
    ratio guard allows only an eight-ULP rate-over-majorant roundoff excess,
    maps it to `1.0`, and rejects material violations before RNG advancement or
    selector mutation. The public executable gate is unchanged: deferred masks
    remain deferred.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: independent deterministic fp64 Warp/NumPy component-sum and bound
    oracles, sparse mixed-scale/non-coincident-maxima fixtures, invalid and
    overflow aggregation cases, eight-versus-nine-ULP ratio cases, selector
    draw/mutation regressions, capped scheduling, and deferred-mask snapshots.

- [ ] **E5-F6-P3:** Validate two-way and four-way single-pass execution with integration tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Execute approved additive masks through one candidate stream, one RNG
    pass, one pair buffer, and one charge-aware merge pass.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: representative and matrix-wide two-way cases, full four-way case,
    bounded aggregate collision rates, single/multi-box sparse populations,
    sorted disjoint pairs, buffer identity, RNG reuse/reset, mass conservation,
    charge conservation, and legacy single-term regression.

- [ ] **E5-F6-P4:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Record the verified additive matrix, required inputs, total-majorant
    proof, single-pass semantics, support limits, and downstream handoff.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, coagulation docstrings, and E5
    plan sections
  - Tests: Markdown link/reference validation plus executable import/signature
    checks where snippets are present.
