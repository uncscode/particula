<!-- TEMPLATE: Replace this entire file with your success criteria -->

Define measurable criteria that determine when this feature is complete and
working correctly. Include both pass/fail checks and quantitative metrics.

**Required elements:**
- Checklist of pass/fail success criteria
- Metrics table with baseline, target, and data source

**Example (E16-F6):**
- [ ] One final PR/MR per manifest completion (idempotent -- no duplicates)
- [ ] PR body includes cumulative implementation summary from all slices
- [ ] Handoff comment explicitly states manual merge and conflict handling required
- [ ] Duplicate PR and comment prevention verified under retry conditions

**Metrics:**
| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Final PRs per manifest | 0 (manual) | 1 (automated) | Platform API |
| Duplicate PR rate | N/A | 0% | Manifest checkpoint logs |
