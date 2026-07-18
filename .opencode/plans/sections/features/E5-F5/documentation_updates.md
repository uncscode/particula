# Documentation Updates

- Completed: `docs/Features/data-containers-and-gpu-foundations.md` documents
  the exact direct ST1956 mechanism configuration, required dissipation
  `[m^2/s^3]` and fluid-density `[kg/m^3]` inputs, scalar/active-device
  `wp.float64` per-box forms, device ownership, and persistent RNG/buffer
  behavior.
- Completed: `docs/Features/Roadmap/data-oriented-gpu.md` records E5-F5's
  ST1956-singleton support fact and defers additive combinations to E5-F6,
  singleton validation/evidence consumption to E5-F7, and the consolidated
  support table/direct example to E5-F9.
- Inspected: `docs/Features/condensation_strategy_system.md` has no shared
  coagulation support table and is out of scope for this delivery.
- Deferred to E5-F9: add or extend a direct low-level example after combination
  and support-matrix boundaries settle. Any later example must keep transfers
  explicit, default to Warp CPU, and show heterogeneous `(n_boxes,)`
  dissipation/fluid-density arrays when useful.
- Completed lifecycle record: the relevant E5-F5 P4 plan records now describe
  the implemented direct contract, ST1956-only scope, device policy, and
  E5-F6/E5-F7/E5-F9 handoffs.
- State prominently in every support table/example that this is the ST1956
  turbulent-shear kernel only. Do not use “DNS parity,” “DNS turbulence,” or
  general turbulence support wording except to mark those capabilities
  unsupported.
- Validate markdown links, import paths, SI units, optional CUDA wording, and
  all snippets. No README or top-level `particula.gpu` export is planned unless
  a later API review explicitly approves it.
