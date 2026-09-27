# Open Questions

- [ ] **Q1 — Entry blocker: Which approved M1 revision freezes the contracts?**
  Consume M1's decisions on species mapping, normalization, API names,
  copy/view behavior and validator placement. They are currently draft review
  questions. Recommendation: accept one reviewed M1 packet, not a second M2
  specification; reopen M1 if a construction requirement is missing.
- [ ] **Q2 — P1 blocker: What minimal temporary transition keeps later-track
  consumers operational?** Existing aggregate/builders and consumers use facade
  shapes. Recommendation: concrete narrowly bounded seam(s), read/write/alias
  rules and consumer inventory approved before P1; no permanent public facade
  or duplicate state. Record M3/M4/M5 migration ownership and M6 deletion proof.
  If no safe bounded transition exists, escalate rather than waive serial gates.
- [ ] **Q3 — P3 gate: Which initialization capabilities and public placements
  are retained?** Seed inventory covers direct mass/radius, PDF/PMF radius-bin,
  resolved mass and sampled lognormal presets. Recommendation: reuse existing
  native builders plus small construction utilities; review actual supported
  consumers before finalizing native module names and obsolete-name disposition.
- [ ] **Q4 — P5 gate: Which existing defaults and random-state behavior must
  native sampled presets preserve?** Recommendation: characterize current
  sampling/defaults, preserve physical distributions and document RNG ownership;
  use controlled fixtures for conversion tests, not new cross-backend replay
  promises. Maintainer/scientific reviewer approves any intentional difference.
- [ ] **Q5 — Handoff governance: Who signs M1 admission and M2 completion?**
  Assign construction/scientific reviewers and M3 receiver. Recommendation:
  named final-revision evidence review; no deadline is supplied by issue #1602.

Fixed, not open choices: three native containers, process-owned physics,
identity-preserving access, rejection atomicity, coordinated replacement,
no automatic GPU rebinding, CPU single-box execution, retained transfer helpers,
migration before removal and strictly serial completed-and-validated tracks.
