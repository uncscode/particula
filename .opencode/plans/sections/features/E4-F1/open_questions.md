# Open Questions

- [x] When thermodynamic configuration is omitted, should
  `condensation_step_gpu()` fail immediately or select a compatibility mode?
  - Resolved 2026-07-13: fail before mutation. Compatibility, if later needed,
    must use an explicitly named static-pressure mode; stale or zero-filled
    vapor pressure is never selected implicitly.
- [x] Should configuration be a dedicated Warp dataclass or keyword-only mode
  and parameter arrays?
  - Resolved 2026-07-13: expose one keyword-only `thermodynamics` sidecar whose
    typed fields hold validated parallel Warp arrays. Keep process configuration
    out of `WarpGasData` and preserve positional compatibility.
- [x] What fixed parameter width should be reserved for future models?
  - Resolved 2026-07-13: reserve one `float64` parameter per species. Constant
    mode consumes the pressure value; Buck consumes no species parameter. Add a
    versioned field or new sidecar contract if a future approved model needs
    more parameters rather than reserving speculative width now.
- [x] Is four-substep orchestration part of E4-F1?
  - Resolved 2026-07-12: No. E4-F1 provides refresh integration for the current
    step; E4-F3 owns production four-substep scheduling and must call refresh at
    each substep.
- [x] Are activity, latent heat, and gas coupling included?
  - Resolved 2026-07-12: No; they belong to E4-F2, E4-F4, and E4-F5.
