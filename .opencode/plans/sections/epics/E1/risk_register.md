<!-- TEMPLATE: Replace this entire file with the risk register -->

List each risk with likelihood, impact, mitigation, and status.

| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|------------|--------|------------|-------|--------|
| Risk description | Low/Medium/High | Low/Medium/High | Mitigation plan | @owner | Open/Mitigated/Closed |

**Example (E17):**
| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|------------|--------|------------|-------|--------|
| Migration script misparses existing plans | Medium | High | Validate via diff comparison against current output | @gorkowski | Mitigated |
| Agent tooling breaks during transition | Medium | Medium | Backward-compatible period where both formats are valid | @gorkowski | Open |
| Generated markdown quality regresses | Medium | High | Snapshot tests comparing generated vs current output | @gorkowski | Open |
| Analytics JSONL grows unbounded | Low | Low | Periodic rotation; summary aggregates replace raw events | @gorkowski | Open |
