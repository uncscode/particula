# Documentation Updates

- Update `docs/Features/data-containers-and-gpu-foundations.md` with authoritative
  charge ownership: dimensionless elementary-charge counts in
  `WarpParticleData.charge`, recipient-add/donor-clear merge semantics, and no
  hidden transfer or sidecar.
- Update `docs/Features/coagulation_strategy_system.md` with the approved GPU
  charged pair-model subset, CPU formula references, tested limits, and the
  distinction between E5-F2 pair helpers and E5-F3 executable sampling.
- Update `AGENTS.md` GPU coagulation notes with charge validation and separate
  per-box mass/charge conservation commands once tests ship.
- Update `.opencode/plans/sections/epics/E5/` and this feature's sections with
  resolved model decisions, phase issue numbers/statuses, evidence links, and
  any deferred variant owner.
- Completed for P1: this feature's structured plan sections record issue #1336,
  the internal-helper/parity-test delivery, and its intentionally unchanged
   public and execution boundaries. No user-facing documentation changed in P1.
- Completed for P4 (#1339): this feature's structured plan sections record the
  private charge-transfer/clear merge semantics and deterministic conservation
  evidence. No README, API reference, or user-facing guide changed because the
  public coagulation API and charged-execution support boundary are unchanged.
- Completed for P5 (#1340): published documentation now identifies charge as
  caller-owned dimensionless elementary-charge counts with fp64/shape/device/
  finiteness preflight and recipient-add/donor-clear bookkeeping. It documents
  the approved helper subset, bounded evidence, and E5-F3 handoff without a
  public charged-execution claim; AGENTS.md records the warning-clean focused
  evidence commands.
- Defer user-facing direct charged execution examples and final support/import
  matrix changes to E5-F9 after E5-F3 and E5-F7 establish execution evidence.
  Do not update README quick-start claims in this foundational feature.
