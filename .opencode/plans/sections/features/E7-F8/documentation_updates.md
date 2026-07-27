# Documentation Updates

- Update `docs/Features/data-containers-and-gpu-foundations.md` with stream
  ownership, stable logical box identity, process namespaces, seed/reset rules,
  state authority, and explicit checkpoint transfer boundaries.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` Track T8 status and record
  same-backend restart guarantees plus cross-backend limitations.
- Update the E7-F4 resident-session documentation/example with root seed,
  logical box IDs, nonterminal checkpoint, fresh-session restart, and explicit
  targeted reset usage; do not introduce per-step inspection or transfer.
- Update Brownian coagulation and wall-loss feature docs to distinguish direct
  kernel convenience RNG from session-managed persistent streams.
- Update `AGENTS.md` quick-reference guidance with focused RNG/restart tests and
  warnings against implicit reseeding or positional box IDs.
- Update these E7-F8 section files when phases ship, including final issue links,
  exact supported schema version, implementation paths, and validation evidence.

Validation: run `mkdocs build --strict` and relevant documentation regression
tests. Example commands must default to Warp CPU and label CUDA as optional.
