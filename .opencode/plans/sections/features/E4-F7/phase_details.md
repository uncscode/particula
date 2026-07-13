# Phase Details

Phase issue creation is intentionally deferred until E4 implementation issues
are generated and scheduled; `TBD` is not an unresolved design decision.

- [ ] **E4-F7-P1:** Publish canonical GPU condensation support contract and guardrails
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Document only the modes, devices, shapes, precision, substeps, buffers, diagnostics, mutation semantics, and unsupported boundaries verified by E4-F1 through E4-F6.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`, `docs/Features/particle-data-migration.md`, `particula/tests/condensation_latent_heat_docs_test.py`
  - Tests: Documentation wording assertions, links, API-name checks, and retained CPU notebook guardrails.

- [ ] **E4-F7-P2:** Extend the explicit-transfer condensation example with regression tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Demonstrate final E4 configuration, caller-owned stable buffers, Warp CPU execution, diagnostics, and explicit restore without hidden host work.
  - Files: `docs/Examples/gpu_direct_kernels_quick_start.py`, `docs/Examples/index.md`, `particula/gpu/tests/gpu_direct_kernels_example_test.py`
  - Tests: Example execution on Warp CPU, exact output, lazy imports, no-Warp behavior, explicit transfer calls, and optional CUDA skip behavior where applicable.

- [ ] **E4-F7-P3:** Publish troubleshooting and focused reproduction commands with documentation tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Give users deterministic failure guidance and commands for the example, contract, stiffness, CPU reference, conservation, and docs suites.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`, `docs/Features/particle-data-migration.md`, `readme.md`, documentation tests
  - Tests: Command/path existence, troubleshooting keyword/constraint assertions, focused suites on Warp CPU, and warning-clean execution.

- [ ] **E4-F7-P4:** Update development documentation and roadmap status
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Link the canonical contract from discovery surfaces and update E4 status only after every dependency exit criterion is evidenced.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`, `docs/Examples/index.md`, `readme.md`, relevant plan sections
  - Tests: Markdown link validation, roadmap wording guardrails, and final focused documentation suite.
