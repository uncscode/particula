# Open Questions

- [ ] What exact concrete-module names should P1 freeze for device configuration,
  scratch, finalized-demand, and diagnostic sidecars?
  - Constraint: names may change, but fields must remain typed, caller-owned,
    fixed-shape, same-device, and identity-preserving when supplied.
- [ ] Does the frozen E6-F7 model require temperature or saturation-gate inputs
  in the first direct GPU API, and if so are they scalar-or-per-box or strictly
  device arrays?
  - Constraint: semantics must match E6-F7, normalization must be explicit, and
    no host-computed fallback may be introduced.
- [ ] Which exact E6-F6 scratch fields and scale bounds are required by the GPU
  nucleation adapter after E6-F6's P1 contract is finalized?
  - Open dependency: consume the shipped E6-F6 contract without duplicating it.
- [ ] Which deterministic CPU/GPU tolerances differ, if any, from the target
  conservation `rtol=1e-12`, `atol=1e-30`?
  - Resolution rule: record fixture-specific evidence and numerical rationale;
    do not relax conservation globally to hide mixed-scale error.
- [x] May the direct step resize arrays or fall back to CPU when capacity or a
  device capability is unavailable?
  - Resolved 2026-07-21: No. It must use E6-F6 or fail before mutation.
- [x] Is a high-level GPU `Runnable` part of E6-F8?
  - Resolved 2026-07-21: No; orchestration and backend selection remain Epic G.
