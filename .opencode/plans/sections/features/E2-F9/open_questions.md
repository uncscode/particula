# E2-F9 Open Questions

## Resolved Answers

- If E2-F2 and E2-F3 have landed, document concrete `EnvironmentData` and
  `WarpEnvironmentData` APIs. If either is still pending, keep environment state
  in a clearly labeled planned/future-work section.
- Create a new guide at `docs/Features/data-containers-and-gpu-foundations.md`
  and link it from `docs/Features/particle-data-migration.md` rather than
  overloading the migration page.
- Use plain Python examples first. Add paired notebooks only if the example needs
  narrative execution output or interactive exploration.
- Authoritative docs validation should be `mkdocs build` or the repository's
  MkDocs validation wrapper when available, plus focused example execution for
  any new example files.
- Demonstrate guarded transfer code that skips cleanly when Warp is unavailable.
  Optional Warp CPU execution may be shown only behind the same availability
  guard.
