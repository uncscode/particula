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

- [ ] **E5-F6-P2:** Implement safe summed majorants and additive pair dispatch with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Sum enabled component rates per candidate and proven component
    majorants per box while enforcing finite non-negative bounded ratios.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/dynamics/coagulation_funcs.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: independent component-sum parity, all-active-pair
    `total_rate <= total_majorant`, zero terms, disparate scales, overflow/
    non-finite guards, and one acceptance draw per proposal.

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
