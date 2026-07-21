# Documentation Updates

- Update `docs/Features/Roadmap/data-oriented-gpu.md` to replace Epic F's
  unscheduled placeholders with E6 and E6-F1 through E6-F9 cross-links, reconcile
  the feature list with the nine delivered tracks, link integrated evidence,
  and record the exit-bar result.
- Update `docs/Features/Roadmap/index.md` with an E6 roadmap inventory and links
  to the complete-process example and relevant support guides; move E6 to
  shipped and Epic G to active only after the closeout gate passes.
- Add `docs/Examples/gpu_complete_process_sequence.py` as the canonical
  explicit-transfer sequence. Document that its fixed call order is illustrative
  and is not a scheduler or high-level runnable.
- Update relevant dilution, wall-loss, nucleation, slot-management, data
  container, condensation, and coagulation feature pages with cross-links to the
  integrated evidence and each component's ownership boundary.
- Update `docs/index.md` so users can discover the example and final E6 support
  contract.
- Update `AGENTS.md` with the final direct-process sequence, focused validation
  commands, transfer rules, RNG/sidecar ownership, and Epic G deferrals.
- Update E6 and E6-F9 plan sections with issue numbers, measured tolerances,
  evidence paths, shipped phase statuses, completion date, and resolved
  questions only when implementation has landed.
- Run documentation link/import/command validation and `adw plans validate`;
  do not publish closeout language while any E6-F1 through E6-F8 dependency is
  incomplete.
