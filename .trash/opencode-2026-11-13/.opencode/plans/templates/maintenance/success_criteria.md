<!-- TEMPLATE: Replace this entire file with success criteria -->

Define when this maintenance area is healthy. Include both checklist items
and a metrics table.

**Required elements:**
- Checklist of health indicators
- Metrics table with baseline, target, and data source

**Example (M23):**
The maintenance area is healthy when:
- [ ] `run_pytest` correctly fails on coverage-gate failures
- [ ] uv-run wrappers return actionable diagnostics
- [ ] `git_operations` continue flows no longer stall after conflict resolution
- [ ] `adw_plans analytics` parity is closed
- [ ] Docs validation defaults avoid notebook-heavy timeouts

**Metrics:**
| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| `run_pytest` coverage fidelity | Can report false PASS | Always returns failure on coverage-gate fail | Bun tests |
| Git continue recovery | Staged conflicts can hang | Advances or returns bounded diagnostic | `git_operations` tests |
| Docs validation runtime | Notebook-heavy can time out | Default skips notebooks unless requested | `build_mkdocs` tests |
