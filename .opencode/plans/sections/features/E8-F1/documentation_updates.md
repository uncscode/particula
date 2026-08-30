# Documentation Updates

- Update `docs/Features/Roadmap/data-oriented-gpu.md` under Epic H to record the
  shipped E8-F1 capability/lifecycle contract, precise invalidation triggers,
  and explicit handoff to E8-F2/E8-F3. State clearly that full-loop captured
  execution and parity are not yet shipped by E8-F1.
- Update `AGENTS.md` with the concrete-only import boundary, legal lifecycle
  transitions, structural compatibility fields, recapture triggers, persistent
  RNG rule, and focused reproduction commands for downstream agents.
- Update relevant E8 parent and E8-F1 plan sections after implementation with
  actual file paths, phase status, decisions, and any revised risks.
- Add Google-style module/class/function docstrings in
  `particula/execution/graph_capture.py`; document which operations are
  metadata-only and which future replay operations may launch writers.
- Extend existing execution export/documentation contract tests when they are
  the repository's source of truth for concrete-only API boundaries.
- Do not create a user-facing `docs/Examples/` graph-capture example in this
  feature. E8-F8 owns the runnable example after E8-F4 validates captured
  execution and E8-F5/E8-F6 establish limits.
- No root README change is expected because E8-F1 adds no public quick-start or
  top-level API.

Validate documentation with `mkdocs build --strict` and any affected docs
contract tests. Keep CUDA instructions pass-or-clean-skip and never imply CPU
graph-capture fallback.
