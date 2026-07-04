# E2-F8 Phase Details

- [ ] **E2-F8-P1:** Audit CPU dynamics container boundaries and baseline docs
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Produce the implementation baseline by confirming condensation,
    coagulation, and docs currently distinguish container shapes from strategy
    execution incompletely.
  - Files: `particula/dynamics/condensation/condensation_strategies.py`,
    `particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py`,
    `docs/Features/particle-data-migration.md`
  - Tests: Add or update the smallest baseline tests needed to capture current
    single-box/box-0 behavior discovered during the audit.

- [ ] **E2-F8-P2:** Clarify single-box and box-0 behavior with focused tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add public-method tests for CPU condensation multi-box rejection and
    CPU coagulation multi-box behavior, preferring explicit unsupported errors
    over silent box-0 mutation where feasible.
  - Files: `particula/dynamics/condensation/tests/condensation_strategies_test.py`,
    `particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py`,
    optional validation helper changes in the corresponding strategy modules.
  - Tests: Multi-box `ParticleData`/`GasData` rejection tests, message-content
    assertions, and any box-0 transitional behavior tests if that behavior is
    intentionally retained.

- [ ] **E2-F8-P3:** Improve unsupported multi-box errors and user documentation
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Finish user-facing and development-doc guidance and align error
    messages with the final tested behavior.
  - Files: `docs/Features/particle-data-migration.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, and strategy modules if error
    wording needs final adjustment.
  - Tests: Documentation examples or assertions that unsupported multi-box CPU
    strategy calls produce clear errors; run focused dynamics tests and docs
    link checks if available.

## Phase Ordering Notes

E2-F8 depends on E2-F1/T1 for container shape conventions. Phase P1 should not
change broad behavior. P2 establishes the executable support boundary. P3 makes
the boundary visible to users and future implementers.
