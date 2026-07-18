# Success Criteria

- [x] The direct GPU step accepts the canonical turbulent-shear-only mechanism
  and preserves omitted-configuration Brownian behavior and return values.
- [x] Dissipation and fluid density are explicit scalar or active-device
  `(n_boxes,)` inputs, use documented SI units, and affect each box independently.
- [x] Missing, zero/negative, non-finite, wrong-shape/dtype/device, and
  unsupported configuration inputs fail before particle/output/RNG mutation.
- [x] Warp fp64 pair values match an independent ST1956 CPU/NumPy oracle under
  declared tolerances across representative scales.
- [x] Every active unordered pair is bounded by the finite non-negative
  turbulent-shear majorant used for trial scheduling and acceptance.
- [x] Turbulent-shear-only execution produces legal sorted/disjoint pairs,
  bounded accepted counts, tight per-box/per-species mass conservation, caller
  buffer identity, and documented persistent RNG reuse/reset behavior.
- [x] Focused Warp CPU tests cover the implemented path; CUDA remains optional
  and skips cleanly when unavailable.
- [x] Documentation says the direct particle-resolved ST1956 singleton only;
  specifies keyword-only positive finite `turbulent_dissipation` (`m^2/s^3`) and
  `fluid_density` (`kg/m^3`) scalar or active-device `wp.float64` `(n_boxes,)`
  forms; and makes no DNS, clustering, inertial-enhancement, general-turbulence,
  or performance claim.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Executable turbulent-shear configurations | 0 | 1 (ST1956-only) | Capability tests |
| Per-box dissipation/density coverage | None | Scalar + heterogeneous `(n_boxes,)` | GPU tests |
| Independent deterministic pair fixtures | 0 | >= 6 across physical scales | Helper/kernel tests |
| Majorant violations over enumerated fixture pairs | N/A | 0 | All-pairs tests |
| Invalid-input state mutations | N/A | 0 | Snapshot tests |
| Species-mass conservation failures | N/A | 0 at declared tight tolerance | Execution tests |
| DNS support claims | 0 | 0 | Documentation review |
