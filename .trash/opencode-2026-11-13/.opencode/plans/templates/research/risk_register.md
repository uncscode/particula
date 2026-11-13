### Purpose
Identify and manage threats to validity, schedule, data integrity, and interpretation quality.

### Required-content checklist
- [ ] Each risk includes likelihood and impact.
- [ ] Mitigation and contingency actions are documented.
- [ ] Trigger conditions for escalation are explicit.
- [ ] Risk ownership is assigned.

### Drafter prompts
- Which threat to validity is most likely to bias conclusions?
- What early-warning indicators should be monitored?
- What is the fastest mitigation that preserves research quality?

### Example
Risk: benchmark leakage from prior tuning data (high impact, medium likelihood). Mitigation: enforce holdout split checksum validation before every run. Trigger: checksum mismatch or unexplained metric jump >5%.
