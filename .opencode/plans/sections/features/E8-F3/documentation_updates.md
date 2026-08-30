# Documentation Updates

- Update `docs/Features/Roadmap/data-oriented-gpu.md` Epic H sections with the
  implemented registry-owned capture resource inventory, setup/reuse boundary,
  persistent RNG handling, and deterministic logical-byte accounting semantics.
- Update `docs/Features/data-containers-and-gpu-foundations.md` if the concrete
  resident ownership table needs process/control/diagnostic resource details;
  keep all new APIs direct-import-only.
- Update `AGENTS.md` only after implementation to summarize validated
  preallocation, identity pinning, and focused/full validation commands.
- Update `.opencode/guides/testing_guide.md` only if new canonical resident
  test or coverage commands are introduced; do not restate transient phase
  commands as permanent policy.
- Cross-reference E8-F1 lifecycle and E8-F2 prepared-enqueue documentation, and
  hand deterministic byte records to E8-F6 without claiming allocator-reserved,
  checkpoint-copy, or future autodiff-tape totals.
- Do not add a user-facing example in this track; E8-F7 owns the complete graph
  capture example and limitation guide.
- Promote phase statuses and update this plan's change log after each shipped
  increment. Validate all links with `mkdocs build --strict`.
