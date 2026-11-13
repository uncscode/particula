<!-- TEMPLATE: Replace this entire file with guidelines and requirements -->

Outline the rules the system should enforce.

**Required subsections:**

### Functional Requirements
Numbered list of concrete requirements.

### Quality Bars
Bullet list of quality standards.

### Constraints
Document rate limits, rollout safety checks, or dependency freezes.

**Example (M23):**

### Functional Requirements
1. `run_pytest` must never report PASS when coverage gating has failed.
2. `git_operations` must advance `rebase --continue` after conflicts are
   resolved, or return a clear next-state diagnostic instead of hanging.
3. `adw_plans` must support `analytics` with wrapper, test, and docs parity.

### Quality Bars
- Wrapper error output must distinguish user-actionable input errors from
  tool, environment, and workflow-state failures.
- All hardening changes must ship with co-located regression tests.
- Default-fast docs validation must reduce timeout risk without hiding the
  explicit full-validation path.

### Constraints
- Preserve existing security posture by preferring typed wrappers over
  generic shell execution.
- Do not weaken CI coverage thresholds.
- Keep output envelopes backward-compatible where possible.
