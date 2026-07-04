# E2-F9 Open Questions

- Have E2-F2 and E2-F3 introduced concrete `EnvironmentData` and
  `WarpEnvironmentData` APIs by the time this feature is implemented, or should
  the guide keep environment state entirely in the planned/future-work section?
- Should the foundation guide be a new page
  `docs/Features/data-containers-and-gpu-foundations.md`, or should it extend
  `docs/Features/particle-data-migration.md` with a clearer GPU section?
- Should examples be plain Python only, paired notebooks, or both?
- What docs validation command should be considered authoritative for this
  repository in the implementation PR?
- Should optional Warp CPU execution be demonstrated, or should examples only
  show guarded transfer code that skips when Warp is unavailable?
