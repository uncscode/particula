# Phase Details

- [ ] **E5-F9-P1:** Publish bounded GPU coagulation support contract with documentation tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Document the shipped mechanisms, configuration, import path, ownership, device policy, unsupported scope, and focused validation commands without broadening claims.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`, `docs/Features/coagulation_strategy_system.md`, `particula/tests/gpu_coagulation_docs_test.py`
  - Tests: Exact support wording, import path, command matrix, deferred-capability ownership, and resolving cross-links.

- [ ] **E5-F9-P2:** Add direct GPU coagulation example with co-located tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Run a small supported particle-resolved configuration on Warp CPU by default with explicit transfers, caller buffers, and persistent RNG.
  - Files: `docs/Examples/gpu_coagulation_direct.py`, `particula/gpu/tests/gpu_coagulation_direct_example_test.py`
  - Tests: CPU-only import/no-Warp branch, lazy runtime imports, deterministic metadata, execution, mutation/conservation checkpoint, buffer identity, and RNG advancement.

- [ ] **E5-F9-P3:** Reconcile roadmap plan IDs and artifact links with validation tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Record E5 and E5-F1 through E5-F9, completed scope, artifacts, and explicit pre-closeout status while keeping all links machine-checked.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`, `docs/Features/Roadmap/index.md`, `particula/tests/gpu_coagulation_docs_test.py`
  - Tests: Complete unique ID matrix, expected artifact labels/targets, status consistency, and local-link resolution.

- [ ] **E5-F9-P4:** Update development documentation and complete dependency-gated epic closeout
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Link all publication artifacts, run the release gate, then and only then mark E5 shipped and Epic F active.
  - Files: `docs/Examples/index.md`, both roadmap files, E5/E5-F9 plan status metadata and sections
  - Tests: E5-F1-F8 shipped-state gate, E5-F9 P1-P3 checks, focused GPU suites, example execution, docs/link validation, and final E5/Epic F status assertions.

P1 and P2 may proceed in parallel. P3 consumes stable artifact paths from P1,
P2, E5-F7, and E5-F8. P4 is strictly last and must remain blocked until every
listed dependency and validation command passes.
