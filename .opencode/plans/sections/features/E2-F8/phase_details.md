# E2-F8 Phase Details

- [x] **E2-F8-P1:** Audit CPU dynamics container boundaries and baseline docs
  - Issue: #1218 | Size: XS | Status: Completed
  - Goal: Produce the implementation baseline by confirming condensation and
    coagulation distinguish container shapes from strategy execution
    incompletely, then lock the current behavior in with focused regressions.
  - Files: `particula/dynamics/condensation/tests/condensation_strategies_test.py`,
    `particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py`
  - Tests: Added a public condensation regression that preserves the existing
    single-box `ValueError`, plus coagulation regressions proving helper-backed
    reads and particle-resolved `step()` mutations were still box-0-only.
  - Docs: General-doc review concluded no user-facing doc edit was needed in
    this audit-only phase; follow-up doc targets remain queued for later phases.

- [x] **E2-F8-P2:** Clarify single-box requirements with focused tests
  - Issue: #1219 | Size: S | Status: Completed
  - Goal: Replace the P1 coagulation box-0 baseline with explicit unsupported
    multi-box CPU errors while keeping condensation public-method coverage
    aligned with its existing single-box guard.
  - Files: `particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py`,
    `particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py`,
    `particula/dynamics/condensation/tests/condensation_strategies_test.py`
  - Tests: Added representative condensation public multi-box rejection
    coverage, replaced coagulation box-0-only regressions with multi-box
    rejection assertions, and retained supported single-box non-regressions.

- [x] **E2-F8-P3:** Improve unsupported multi-box errors and user documentation
  - Issue: #1220 | Size: S | Status: Completed
  - Goal: Publish the tested CPU support boundary in user-facing docs without
    expanding runtime behavior.
  - Files: `docs/Features/particle-data-migration.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`
  - Result: The migration guide now acts as the canonical support contract,
    including a compact CPU support table, a supported `n_boxes == 1` example,
    caller-managed per-box loop guidance, and a troubleshooting cross-reference.
    The roadmap wording was narrowed so container compatibility is not read as
    shipped CPU multi-box execution.
  - Tests: Docs-only/manual validation; no strategy-module or behavior changes
    were needed in this phase.

## Phase Ordering Notes

- P1 depends on E2-F1 for the accepted container shape vocabulary and should
  stay audit-only so sibling schema work is not reopened here.
- P2 followed P1 and the accepted E2-F2 environment wording so explicit
  unsupported multi-box errors now describe the same CPU-versus-environment
  boundary used elsewhere in the epic.
- P3 followed the tested rejection behavior from P2 so docs now publish the
  exact unsupported-path contract rather than an aspirational boundary.
- E2-F9 should document the support boundary only after P3 lands, which keeps
  user-facing guidance downstream of the executable CPU error contract.
