# Phase Details

- [x] **E5-F1-P1:** Define mechanism configuration and support validation with unit tests
  - Issue: #1331 | Size: S | Status: Completed
  - Goal: Add canonical mechanism identifiers, a frozen configuration, a
    resolved execution mask, and structural/capability validators while
    preserving the omitted-configuration Brownian default.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Completed: Added host-only configuration, resolver, capability validator,
    and co-located resolver/capability tests. Fixed masks are Brownian `1`,
    charged hard-sphere `2`, SP2016 `4`, and ST1956 `8`; only Brownian is
    executable. `coagulation_step_gpu` signature and GPU runtime are unchanged.

- [x] **E5-F1-P2:** Introduce additive pair-rate and single-pass sampling interfaces with unit tests
  - Issue: #1332 | Size: S | Status: Completed
  - Goal: Route Brownian through shared pair-rate and majorant dispatch whose
    contract sums enabled non-negative terms before one acceptance decision.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/dynamics/coagulation_funcs.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Completed: Added private additive mask-dispatch helpers for sanitized,
    finite-positive Brownian pair rates and majorants. The kernel computes one
    total rate/majorant and uses one acceptance draw for each valid candidate;
    reserved flags are no-ops and unsafe inputs skip before mutation. Co-located
    unit, fixed-seed deterministic, and marked stochastic tests cover helper
    parity, synthetic addition, one-draw behavior, guards, selector validity,
    and mass conservation.

- [x] **E5-F1-P3:** Enforce pre-mutation particle-resolved support boundaries with integration tests
  - Issue: #1333 | Size: S | Status: Completed
  - Goal: Integrate keyword-only mechanism configuration into
    `coagulation_step_gpu` and prove every unsupported request fails before
    state, output buffers, allocations, or RNG launches are changed.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Completed: Added host-only configuration preflight as the first runtime
    action, with `None` selecting Brownian and the resolved mask passed to the
    existing launch. Rejected wrong-type, structural, distribution, and reserved
    configurations fail before runtime access or mutation. Co-located Warp CPU
    integration tests cover failure atomicity, ordering sentinels, and equal-seed
    omitted-versus-explicit Brownian equivalence across environment source forms.

- [ ] **E5-F1-P4:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document extension rules, supported/reserved mechanism matrix,
    additive sampling semantics, and the particle-resolved-only boundary.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, relevant docstrings, and E5
    plan sections.
  - Tests: markdown link/reference validation and doctest/import checks where
    configuration examples are executable.
