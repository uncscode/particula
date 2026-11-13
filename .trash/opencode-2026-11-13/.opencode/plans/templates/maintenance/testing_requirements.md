<!-- TEMPLATE: Replace this entire file with testing requirements -->

Every phase must ship with co-located tests. Coverage thresholds must never
be lowered.

**Test Coverage Policy (include verbatim):**
1. Test coverage thresholds must NEVER be lowered
2. Each phase must include self-contained tests
3. Tests are committed in the same PR as the implementation
4. Test files use `*_test.py` suffix in module-level `tests/` directories
5. Minimum 80% coverage (configured in `pyproject.toml`)

**Maintenance-Specific Testing:**
- **Regression Tests:** Ensure fixes don't break existing functionality
- **Health Checks:** Validate monitoring/audit scripts work correctly
- **Wrapper contract tests:** Verify status classification and error envelopes
- **Parity tests:** Wrapper surface changes must ship with matching tests

**Example (M23):**
- Wrapper contract tests: Verify status classification, error envelopes, and
  env-sanitized invocation behavior
- Workflow recovery tests: Simulate pruned worktrees, staged conflict
  resolution, and other partial-state recovery paths
- Docs-validation tests: Confirm fast default mode is explicit and that the
  notebook-inclusive path remains available
