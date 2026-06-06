### Purpose
Capture upstream inputs and downstream consumers that constrain sequencing, reproducibility, and delivery risk.

### Required-content checklist
- [ ] Upstream artifacts/services are listed with owners.
- [ ] Downstream consumers and handoff expectations are documented.
- [ ] Critical path sequencing assumptions are explicit.
- [ ] Contingency plans for blocked dependencies are defined.

### Drafter prompts
- Which dependency has the highest schedule risk?
- What fallback path preserves experiment continuity if blocked?
- Which dependency assumptions must be validated early?

### Example
Upstream: feature-store snapshot publication by Data Platform (owner: DP-oncall). Downstream: model-governance review package. If snapshot is delayed >48h, use prior validated snapshot and record variance impact in evaluation notes.
