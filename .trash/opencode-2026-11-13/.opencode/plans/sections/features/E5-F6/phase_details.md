# Phase Details

- [x] **E5-F6-P1:** Define additive combination matrix and preflight validation with unit tests
  - Issue: #1357 | Size: S | Status: Implemented
  - Delivered: A private immutable recognition table accepts four singleton,
    six two-way, and one four-way mask. Non-turbulent deferred mask `7` is
    capability-rejected before particle shape/device metadata access or
    enabled-term validation. Turbulent deferred masks `11`, `13`, and `14`
    source-read that metadata and validate enabled terms before rejecting.
    Recognition is separate from executable support: valid deferred masks raise
    the stable deferred-execution error before downstream normalization,
    allocation, RNG setup, launch, or mutation.
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

- [x] **E5-F6-P3:** Ship approved additive single-pass direct-kernel execution
  - Issue: shipped | Size: S | Status: Implemented
  - Delivered: Masks `1`, `2`, `4`, `8`, `3`, `5`, `6`, `9`, `10`, `12`, and
    `15` execute through one candidate stream, one RNG stream, one pair buffer,
    and one charge-aware merge/apply pass; three-way masks `7`, `11`, `13`, and
    `14` remain deferred. Mask `7` is capability-rejected before metadata or
    enabled-term checks; turbulent masks `11`, `13`, and `14` reject after them.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: Existing public-path, conservation, ownership, selector,
    deferred-mask, and persistent-RNG evidence in `coagulation_test.py`.

- [x] **E5-F6-P4:** Update development documentation
  - Issue: #1360 | Size: XS | Status: Implemented
  - Goal: Record the verified additive matrix, required inputs, total-majorant
    proof, single-pass semantics, support limits, and downstream handoff.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, coagulation docstrings, and E5
    plan sections
  - Validation: Documentation work is complete; source/signature and Markdown
    inspection, ruff, and the focused existing coagulation suite are confirmed
    final checks. This does not claim E5-F7 release validation or E5-F9 example
    delivery.
