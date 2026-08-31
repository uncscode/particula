# Documentation Updates

- Issue #1549 made no user-documentation changes. The resident binding and
  scheduler gate remain direct-module-only implementation seams, not a public
  API or user workflow.
- Update relevant E8 parent and E8-F1 plan sections after implementation with
  actual file paths, phase status, decisions, and any revised risks.
- Extend existing execution export/documentation contract tests when they are
  the repository's source of truth for concrete-only API boundaries.
- Do not create a user-facing `docs/Examples/` graph-capture example in this
  feature. E8-F8 owns the runnable example after E8-F4 validates captured
  execution and E8-F5/E8-F6 establish limits.
- No root README change is expected because E8-F1 adds no public quick-start or
  top-level API.

No documentation build is required for this implementation-only phase. Preserve
the no-CPU-fallback boundary through implementation and export tests.
