# Documentation Updates

- Update `docs/Features/data-containers-and-gpu-foundations.md` with the direct
  turbulent-shear mechanism configuration, required dissipation `[m^2/s^3]`
  and fluid-density `[kg/m^3]` inputs, scalar/per-box forms, device ownership,
  and persistent RNG/buffer behavior.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` with E5-F5's completed
  ST1956-only support fact and handoffs to E5-F6/F7/F9.
- Update `docs/Features/condensation_strategy_system.md` only if its shared
  direct-kernel support table covers coagulation mechanisms; avoid unrelated
  condensation-contract edits.
- Add or extend a direct low-level example only after the final concrete API is
  stable. Keep transfers explicit, default to Warp CPU, and show heterogeneous
  `(n_boxes,)` dissipation/fluid-density arrays when useful.
- Update `.opencode/plans/sections/features/E5-F5/*.md` phase/status facts as
  implementation ships and cross-reference parent E5 and downstream tracks.
- State prominently in every support table/example that this is the ST1956
  turbulent-shear kernel only. Do not use “DNS parity,” “DNS turbulence,” or
  general turbulence support wording except to mark those capabilities
  unsupported.
- Validate markdown links, import paths, SI units, optional CUDA wording, and
  all snippets. No README or top-level `particula.gpu` export is planned unless
  a later API review explicitly approves it.
