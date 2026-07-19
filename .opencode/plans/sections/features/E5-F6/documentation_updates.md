# Documentation Updates

- [x] Update the `coagulation_step_gpu` and mechanism-configuration docstrings in
  `particula/gpu/kernels/coagulation.py` with the approved matrix, canonical
  names, required per-mechanism inputs, summed-majorant rule, and one-pass RNG
  semantics.
- [x] Update `docs/Features/data-containers-and-gpu-foundations.md` with caller
  ownership of additive inputs, one output pair buffer, one persistent per-box
  RNG stream, and charge-aware merge behavior.
- [x] Update `docs/Features/Roadmap/data-oriented-gpu.md` with E5-F6 status,
  executable singleton masks `1`, `2`, `4`, `8`, two-way masks `3`, `5`, `6`,
  `9`, `10`, `12`, four-way mask `15`, rejected three-way masks `7`, `11`,
  `13`, `14`, and the E5-F7/E5-F9 handoff.
- [x] Record verified support boundaries in the E5 epic and E5-F6 plan sections;
  do not mark E5-F7's cross-mechanism matrix or E5-F9's final example complete
  from this track.
- [x] Provide E5-F9 with concise example requirements: direct low-level import,
  explicit transfers, explicit turbulent inputs when enabled, Warp CPU default,
  optional CUDA, and no high-level runnable or performance claim.
- [x] Retain `AGENTS.md` without change: its shipped low-level usage contract is
  sufficient for the repository quick reference; no new agent/workflow is added.

Documentation must distinguish additive pair-rate semantics from independent
sequential mechanism steps and must state that `sum(term_majorants)` is safe but
potentially conservative. P4 records the completed documentation reconciliation;
the retained source/signature, Markdown, ruff, and focused-suite checks are its
final validation plan, not E5-F7 release evidence.
