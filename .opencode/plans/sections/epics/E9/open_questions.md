# Open Questions

- [ ] **M1 gate: exact ordered species mapping.** Must all containers share a
  full gas-width order, or is an explicit process-owned partitioning-lane map
  required? `gas_data.py:65–80` provides names/mask, while environment species
  lanes have no names (`environment_data.py:30–43`). Recommend documenting and
  testing an explicit ordered mapping that preserves current schemas and all
  nonpartitioning gas; do not infer identity from equal shapes alone. M1 owner
  and scientific reviewer must select the precise representation before M2.
- [ ] **M1/M2 gate: public spelling and mutation rules.** Select constructor,
  individual getter/setter and coordinated replacement names, plus field-level
  copy/view semantics. Recommend one consistent minimal API with preflight
  before publication; the identity/no-partial-replacement contract is fixed.
- [ ] **M2 gate: supported initialization inventory.** Which current builder
  and preset capabilities need native equivalents, and where do they live?
  Inventory actual consumers and supported examples before removing names;
  preserve useful behavior without recreating facade-based factories.
- [ ] **M2 gate: temporary compatibility shape.** Which minimal bridge keeps
  not-yet-migrated consumers operational while serial tracks proceed?
  Recommend concrete, explicitly inventoried temporary seams with consumer
  lists and M6 deletion proof, not a new permanent public facade.
- [ ] **M5 gate: supported example inventory and historical exceptions.**
  Confirm every runnable/notebook in scope and explicitly label retained
  historical before/after snippets. A new quick start alone is insufficient.
- [ ] **Release governance: named reviewers and scheduling.** Maintainer to
  assign track/release reviewers and dates; v0.3.0 has no fixed deadline.
- [ ] **Plan tooling: accept epic milestones as program phases?** The actual
  schema rejects epic `add-phase`; six milestones are stored instead. Recommend
  retaining supported milestones and detailed phases on maintenance children,
  rather than inventing unsupported phase fields or IDs. Orchestrator must
  acknowledge this representation in its final report.

Already fixed by issue #1602: serial execution, migration before deletion,
single-box CPU execution, three unchanged container names, process-owned
physics, retained CPU↔Warp transfers and no automatic GPU/resident rebinding.
These are not open scope choices.
