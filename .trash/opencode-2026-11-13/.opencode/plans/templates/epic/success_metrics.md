<!-- TEMPLATE: Replace this entire file with success metrics -->

Describe how success is measured across the full program. Tie metrics back
to the metadata block for quick reference.

**Required elements:**
- Measurable criteria (checklist format)
- Quantitative metrics where applicable

**Example (E17):**
- All 134+ existing plans have valid JSON records passing schema validation
- Generated markdown indexes are equivalent to current hand-maintained indexes
- `adw plans validate` passes on CI with zero errors
- `adw plans list` returns correct results for type/lifecycle/parent filters
- Runtime analytics capture cost and duration for at least 10 workflow runs
- No plan file moves required on status changes
- Phase status auto-updates within 30 seconds of workflow completion
- Plan auto-promotes to Shipped when all phases complete (no human action)
