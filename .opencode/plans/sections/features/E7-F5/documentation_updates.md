# Documentation Updates

- Add a user-facing scheduler/process-order section to the appropriate
  `docs/Features/` execution or GPU-resident simulation guide, including the
  canonical dependency diagram and supported process matrix.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` Track T5 status and evidence
  without claiming E7-F7 transport, E7-F8 final RNG semantics, or E7-F9 closeout.
- Update `docs/Features/data-containers-and-gpu-foundations.md` with resident
  state authority, explicit checkpoint boundaries, and no-hidden-transfer rules.
- Extend a scheduler-focused example only when the dependent public APIs exist;
  keep `docs/Examples/gpu_complete_process_sequence.py` labeled illustrative.
- Update `AGENTS.md` with canonical process order, focused validation commands,
  environment/gas freshness rules, and explicit unsupported boundaries.
- Update E7 and E7-F5 plan sections with shipped phase status and issue links.
- Run documentation regressions and `mkdocs build --strict`; if a paired notebook
  is touched, edit its Jupytext `.py`, sync, execute, and commit both files.
