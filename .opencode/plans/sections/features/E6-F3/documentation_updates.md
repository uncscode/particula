# Documentation Updates

- Update `AGENTS.md` with the supported neutral GPU wall-loss geometries, direct
  import, configuration rules, persistent RNG initialize-once/reuse pattern,
  exact fixed-slot clearing contract, and focused test commands.
- Update the appropriate `docs/Features/` GPU/process page with CPU-reference
  formulas and citations, SI units, explicit transfer/ownership boundaries,
  scalar/per-box/environment inputs, asynchronous mutation boundary, and
  deterministic versus statistical parity claims.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` only to reflect E6-F3's
  shipped neutral capability and retain Epic G deferrals; final plan-ID/status
  cross-linking remains owned by E6-F9.
- Add or update a low-level snippet under `docs/Examples/` only if E6-F9's
  integrated example cannot serve as the canonical usage. The snippet must
  transfer explicitly, default to Warp CPU, initialize caller-owned RNG once,
  reuse it across steps, and avoid high-level GPU runnable claims.
- Update `.opencode/guides/testing_guide.md` if a reusable stochastic survival
  validation convention is introduced, including confidence bounds and CUDA
  skip behavior.
- Update all E6-F3 plan sections after implementation with final symbol names,
  file paths, recorded tolerances, issue numbers, and shipped phase statuses.
- Cross-reference parent E6, downstream E6-F4 charged/fallback work, and E6-F9
  closeout. Explicitly list charged physics, scheduler/backend integration,
  graph capture, dynamic slots, and performance claims as deferred.

No README change is required unless the direct kernel becomes part of a public
quick-start surface. Documentation must be reviewed against implementation and
validated for links/imports in P7.
