### Purpose
Specify how evidence will be evaluated, including quantitative metrics, qualitative checks, and decision checkpoints.

### Required-content checklist
- [ ] Primary and secondary metrics are defined with thresholds.
- [ ] Validation datasets or scenarios are listed.
- [ ] Evaluation cadence and checkpoints are scheduled.
- [ ] Criteria for escalation, continuation, or stop are explicit.

### Drafter prompts
- Which metrics represent user or system value most directly?
- How will metric regressions be interpreted against uncertainty bounds?
- At what checkpoint is a go/no-go recommendation made?

### Example
Evaluate weekly against held-out benchmark set with thresholds of `>=0.82` macro-F1 and `<=120ms` p95 latency; trigger redesign if two consecutive checkpoints miss both targets.
