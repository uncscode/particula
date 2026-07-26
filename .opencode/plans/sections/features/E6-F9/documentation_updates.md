# Documentation Updates

- Update `docs/Features/Roadmap/data-oriented-gpu.md` to replace Epic F's
  unscheduled placeholders with E6 and E6-F1 through E6-F9 cross-links, reconcile
  the feature list with the nine delivered tracks, link integrated evidence,
  and record the exit-bar result.
- Update `docs/Features/Roadmap/index.md` with an E6 roadmap inventory and links
  to the complete-process example and relevant support guides; move E6 to
  shipped and Epic G to active only after the closeout gate passes.
- [x] Added `docs/Examples/gpu_complete_process_sequence.py` as the canonical
  explicit-transfer sequence (#1448). Its module and stable output document one
  setup transfer, five ordered direct calls, one checkpoint restore,
  caller-owned sidecars/RNG, deterministic no-Warp behavior, and that the order
  is illustrative rather than a scheduler or high-level runnable.
- [x] Updated relevant dilution, wall-loss, nucleation, slot-management, data
  container, condensation, and coagulation feature pages with cross-links to the
  integrated evidence and each component's ownership boundary.
- [x] Updated `docs/index.md` so users can discover the example and final E6 support
  contract.
- [x] Updated `AGENTS.md` with the final direct-process sequence, focused validation
  commands, transfer rules, RNG/sidecar ownership, and Epic G deferrals.
- [x] Updated E6-F9 plan sections with #1448 P3 evidence and resolved transfer,
  no-Warp, and fallback-boundary questions.
- [x] Ran documentation link/import/command validation and `adw plans validate`
  for #1449. The closeout projection remains blocked while E6-F2, E6-F5, E6-F6,
  and E6-F8 are incomplete.
