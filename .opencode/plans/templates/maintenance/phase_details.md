<!-- TEMPLATE: Replace this entire file with your phase checklist -->

List every phase as a checklist item. Each phase should be one reviewable PR
(~100 LOC of production code). Pack related work into a single phase rather
than splitting too finely -- only split when a plan exceeds 15-20 phases.

The final phase should always be "Update development documentation".

**Format per phase:**
- [ ] **{PLAN_ID}-P{n}:** {Phase title}
  - Issue: TBD | Size: {XS|S|M} | Status: Not Started
  - Goal: {One sentence describing the deliverable}
  - Implementation: {Brief description of what to change}
  - Tests: {What tests to add}

**Example (M23):**
- [ ] **M23-P1:** Fix coverage-failure classification in pytest wrapper
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Eliminate false PASS states when pytest coverage gating fails
  - Implementation: Update `.opencode/tools/run_pytest.ts` so coverage threshold
    failures are always treated as failed validation
  - Tests: Add bun tests proving coverage-fail-under output cannot produce
    `VALIDATION: PASSED`

- [ ] **M23-P7:** Add `adw_plans analytics` wrapper parity
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Expose `adw plans analytics` CLI safely through the wrapper
  - Implementation: Add `analytics` to COMMANDS tuple in `adw_plans.ts`,
    add timeout entry, handle `--json` flag
  - Tests: Update rejection test, add positive dispatch tests

- [ ] **M23-P12:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Capture new wrapper contracts and operator guidance
