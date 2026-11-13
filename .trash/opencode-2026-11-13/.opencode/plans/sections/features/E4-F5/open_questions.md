# Open Questions

- [x] What concentration/volume convention maps particle mass to gas kg/m3?
  - Resolved 2026-07-13: `ParticleData.concentration` is represented count.
    Normalize it as `concentration / volume`; gas concentration is kg/m3.
    Convert aggregate transfer back to per-particle mass using that normalized
    concentration, matching the CPU reference.
- [x] Which deterministic reduction implementation is required?
  - Resolved 2026-07-13: use a fixed-shape two-pass box/species reduction in
    deterministic particle-index order. First aggregate positive demand and
    negative release, compute one inventory scale, then apply scaled transfer.
    Avoid cross-particle atomics.
- [x] How is new scratch exposed?
  - Resolved 2026-07-13: extend E4-F3's keyword-only typed sidecar with
    `(n_boxes, n_species)` demand, release, and scale buffers. Preserve public
    positional compatibility.
- [x] What tolerances separate conservation from physics parity?
  - Resolved 2026-07-13: bookkeeping conservation uses `rtol=1e-12` and
    `atol=max(1e-18, scale * eps)`. Physics parity starts at `rtol=1e-10` with
    its independently scale-derived `atol`.
- [x] Where do invalid nonbinary `int32` partition values fail?
  - Resolved 2026-07-14 by issue #1302: validate dtype/shape/device in call
    preflight and inspect manually constructed Warp masks with a private
    status-only device validation launch/readback. Nonbinary values raise
    `ValueError` before mutable work; conversion restoration also rejects
     nonbinary values.
- [x] Can P2 finalize inventory transfer without changing the public contract?
  - Resolved 2026-07-14 by issue #1303: yes. Keep finalization private and
    direct-test-only, require an already P1-gated proposal, and leave
    `condensation_step_gpu()`, `gas.concentration`, public return semantics,
    and energy handling unchanged. P3--P4 own public orchestration.
- [x] How does the public path handle four-substep gas coupling and failure?
  - Resolved 2026-07-14 by issue #1304: each successful cycle finalizes,
    applies, accumulates, and concentration-weights the same transfer into gas;
    later proposals read the updated gas. Aggregate preflight is atomic, but a
     later fresh-proposal failure does not roll back earlier completed cycles.
- [x] Does the shipped public path conserve concentration-weighted inventory
  separately from CPU-oracle physics parity?
  - Resolved 2026-07-14 by issue #1305: yes. Regression-only CPU integration
    coverage accounts for H2O and NH4HSO4 independently and requires exact N2
    invariance. A deterministic fp64 two-box public-hook case checks per-box/
    per-species particle-plus-gas inventory at `rtol=1e-12, atol=1e-30`
    separately from CPU-oracle particle/gas parity at `rtol=2e-10, atol=1e-30`,
    including uptake, evaporation, disabled partitioning, zero gas, and
    zero-concentration slots.

Diagnostics requested: none. Issue #1272's production-hook and conservation
gate is satisfied; broader E4 production claims remain separately gated.
