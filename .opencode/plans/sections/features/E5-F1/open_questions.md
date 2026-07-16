# Open Questions

- [x] Which CPU charged variants receive stable first-release identifiers?
  - Resolved 2026-07-16: expose only `charged_hard_sphere`. Keep Dyachkov 2007,
    Gatti 2008, Gopalakrishnan 2012, and Chahl 2019 unavailable until each has a
    separately approved parity contract. Do not expose a generic `charged` name.
- [x] Should `CoagulationMechanismConfig` be exported from
  `particula.gpu.kernels`, or remain concrete-module-only?
  - Resolved 2026-07-16: keep it concrete-module-only at
    `particula.gpu.kernels.coagulation`, matching specialized low-level
    sidecars. Only `coagulation_step_gpu` remains package-exported through E5.
- [x] Should user ordering be rejected unless canonical or accepted and
  normalized to canonical order?
  - Resolved 2026-07-16: accept any unique ordering and normalize to
    `brownian`, `charged_hard_sphere`, `sedimentation_sp2016`, then
    `turbulent_shear_st1956`, with bits `1`, `2`, `4`, and `8`. Reject duplicate,
    empty, unknown, and unsupported sets.
- [x] How should later scalar/per-box dissipation and fluid-density inputs be
  grouped without coupling unrelated mechanisms?
  - Resolved 2026-07-16: add keyword-only `turbulent_dissipation` and
    `fluid_density` call arguments only in E5-F5. Each accepts a finite scalar or
    active-device `(n_boxes,)` fp64 array. Require both only for ST1956 and reject
    either as excess input when ST1956 is disabled.
- [x] Should invalid device-computed rate/majorant values set a diagnostic
  buffer or simply suppress acceptance after host preflight?
  - Resolved 2026-07-16: use an internal per-box `wp.int32` status buffer and a
    documented validation readback before RNG initialization or mutable
    launches. Raise on non-finite, negative, or material bound violations. Do
    not expose debug buffers or silently suppress invalid physics.

Classifier diagnostics: none.
