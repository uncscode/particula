<!-- TEMPLATE: Replace this entire file with your testing strategy -->

Every phase must ship with co-located tests. Coverage thresholds must never
be lowered.

**Required elements:**
- Per-phase testing approach (what kind of tests for each phase)
- Test file locations following `*_test.py` convention
- Coverage impact assessment

**Test Coverage Policy:**
1. Test coverage thresholds must NEVER be lowered
2. Each phase must include self-contained tests
3. Tests are committed in the same PR as the implementation
4. Test files use `*_test.py` suffix in module-level `tests/` directories
5. Minimum 80% coverage (configured in `pyproject.toml`)

**Testing approach per phase:**
- **Unit Tests:** Test isolated functions/classes with mocked dependencies
- **Integration Tests:** Test component interactions
- **Regression Tests:** Ensure changes don't break existing behavior

**Example (E16-F6):**
- P1: Workflow/agent definition tests (schema validation, instruction parsing)
- P2: Final PR/MR creation and idempotency tests (mock platform router)
- P3: Final comment/guardrail tests (content assertions, duplicate prevention)
- P4: Docs validation (link checking, reference verification)
