# Open Questions

- [ ] When thermodynamic configuration is omitted, should
  `condensation_step_gpu()` fail immediately (the issue #1272 preference) or
  require an explicit named legacy/static-pressure mode for compatibility?
  Silent reuse of stale/zero-filled pressure is not acceptable.
- [ ] Should configuration be a dedicated Warp dataclass or keyword-only mode
  and parameter arrays? Decide in P1 based on existing typed-container patterns;
  do not extend `WarpGasData` with process configuration.
- [ ] What fixed parameter width should be reserved for future models? Keep P1
  limited to the minimum needed by constant and Buck while preserving a stable
  shape contract.
- [x] Is four-substep orchestration part of E4-F1?
  - Resolved 2026-07-12: No. E4-F1 provides refresh integration for the current
    step; E4-F3 owns production four-substep scheduling and must call refresh at
    each substep.
- [x] Are activity, latent heat, and gas coupling included?
  - Resolved 2026-07-12: No; they belong to E4-F2, E4-F4, and E4-F5.
