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
    single-box behavior and any unsupported multi-box paths discovered during
    the audit.

- [ ] **E2-F8-P2:** Clarify single-box requirements with focused tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add public-method tests for CPU condensation multi-box rejection and
    CPU coagulation multi-box rejection, requiring explicit unsupported errors
    instead of silent box-0 mutation.
  - Files: `particula/dynamics/condensation/tests/condensation_strategies_test.py`,
    `particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py`,
    optional validation helper changes in the corresponding strategy modules.
  - Tests: Multi-box `ParticleData`/`GasData` rejection tests and
    message-content assertions for explicit unsupported multi-box errors.

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

- P1 depends on E2-F1 for the accepted container shape vocabulary and should
  stay audit-only so sibling schema work is not reopened here.
- P2 should follow P1 and the accepted E2-F2 environment wording so explicit
  unsupported multi-box errors describe the same CPU-versus-environment boundary
  used elsewhere in the epic.
- P3 should follow the tested rejection behavior from P2 so docs publish the
  exact unsupported-path contract rather than an aspirational boundary.
- E2-F9 should document the support boundary only after P3 lands, which keeps
  user-facing guidance downstream of the executable CPU error contract.
