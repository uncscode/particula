# Phase Details

- [x] **E5-F9-P1:** Publish bounded GPU coagulation support contract with documentation tests
  - Issue: #1372 | Size: S | Status: Shipped
  - Goal: Document the shipped mechanisms, configuration, import path, ownership, device policy, unsupported scope, and focused validation commands without broadening claims.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`, `docs/Features/coagulation_strategy_system.md`, `particula/tests/gpu_coagulation_docs_test.py`
  - Delivered: canonical foundations and CPU strategy guides now publish the
    executable/deferred direct particle-resolved masks, caller ownership,
    preflight order, optional CUDA policy, and explicit exclusions.
  - Tests: `particula/tests/gpu_coagulation_docs_test.py` is stdlib-only and
    checks exact support wording, import path, command matrix,
    deferred-capability ownership, and resolving cross-links.

- [x] **E5-F9-P2:** Add direct GPU coagulation example with co-located tests
  - Issue: #1373 | Size: S | Status: Shipped
  - Goal: Run a small supported particle-resolved configuration on Warp CPU by default with explicit transfers, caller buffers, and persistent RNG.
  - Files: `docs/Examples/gpu_coagulation_direct.py`, `particula/gpu/tests/gpu_coagulation_direct_example_test.py`
  - Delivered: The standalone example explicitly converts CPU `ParticleData`,
    invokes Brownian particle-resolved `coagulation_step_gpu` exactly twice with
    shared collision/RNG sidecars (initializing the RNG only on call one), and
    restores CPU state only after both calls succeed. The disabled route is lazy
    and has no fallback; `docs/index.md` links the bounded route.
  - Tests: CPU-only import/no-Warp branch, exact lazy runtime import order,
    deterministic metadata, sidecar identity and RNG reuse, upstream failure
    propagation, Warp CPU conservation/pair invariants and RNG advancement, and
    zero/one-active-slot cases.

- [x] **E5-F9-P3:** Reconcile roadmap plan IDs and artifact links with validation tests
  - Issue: #1374 | Size: S | Status: Shipped | Completed: 2026-07-20
  - Goal: Record E5 and E5-F1 through E5-F9, canonical statuses, artifacts, and explicit pre-closeout status while keeping all links machine-checked.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`, `docs/Features/Roadmap/index.md`, `particula/tests/gpu_coagulation_docs_test.py`
  - Documented implementation notes: Both roadmap records contain the same
    unique inventory and three artifact links. This phase completed on
    2026-07-20 before the rerun P4 closeout recorded below.
  - Regression coverage: Hardware-free documentation coverage checks the complete unique ID
    matrix, exact artifact labels/targets, record equality, status consistency,
    anti-duplication boundaries, and local-link resolution.

- [x] **E5-F9-P4:** Update development documentation and complete dependency-gated epic closeout
  - Issue: #1375 | Size: S | Status: Shipped | Completed: 2026-07-20
  - Recorded: P4 closeout was rerun after P3 on 2026-07-20. It replaces the
    superseded 2026-07-19 closeout projection and synchronizes E5/E5-F9 as
    shipped with Epic F active.
  - Evidence (all exit 0 and warning-clean; marker-selected runs used Warp CPU):
    ```text
    pytest particula/tests/gpu_coagulation_docs_test.py -q -Werror  # passed
    pytest particula/gpu/tests/gpu_coagulation_direct_example_test.py -q -Werror  # passed
    pytest particula/gpu/kernels/tests/coagulation_validation_test.py -q -m "warp and gpu_parity" -Werror  # passed
    pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -m "warp and stochastic and not cuda" -Werror  # passed
    pytest particula/gpu/kernels/tests/coagulation_test.py -q -Werror  # passed
    python3 .opencode/tools/run_pytest.py  # passed
    python3 .opencode/tools/run_linters.py  # passed
    ```
    The docs test was rerun after the mutating lint workflow.

P1 and P2 may proceed in parallel. P3 consumes stable artifact paths from P1,
P2, E5-F7, and E5-F8. This P4 rerun follows P3 and records the authoritative,
dependency-ordered closeout evidence.
