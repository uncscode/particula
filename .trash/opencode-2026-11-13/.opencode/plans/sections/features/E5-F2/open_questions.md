# Open Questions

- [x] Which CPU charged variants are approved for the first GPU support matrix?
  - Resolved 2026-07-16: approve only the complete hard-sphere system-state
    model as `charged_hard_sphere`. It consumes existing radius, mass, signed
    finite elementary-charge counts, temperature, and pressure, with the CPU
    repulsive-potential floor of `-200`. The other four CPU variants are deferred.
- [x] What exact `rtol`/`atol` applies to each approved pair model and extreme
  repulsive fixture?
  - Resolved 2026-07-16: positive finite hard-sphere pair rates use
    `rtol=1e-6, atol=0`. Neutral-limit and extreme-repulsion zeros are asserted
    exactly with `0.0`.
- [x] Should invalid duplicate recipient indices in a caller-supplied collision
  buffer be defensively ignored, rejected, or remain a documented private-kernel
  precondition?
  - Resolved 2026-07-16: disjoint recipients remain a documented private apply-
    kernel precondition. The production selector overwrites the work buffer and
    emits disjoint pairs, so no O(n²) normal-path scan is added. A future public
    prepopulated-pair boundary must reject duplicates before launch.
- [x] Should charge finite-value validation reuse a shared active-device helper
  or add a coagulation-local helper with non-positive values allowed?
  - Resolved 2026-07-16: add a coagulation-local active-device finite validator
    because signed and zero charges are valid. Validate shape, fp64 dtype,
    device, and finite contents through a compact status readback before RNG or
    mutation, without copying the full charge array to the host.
- [x] Does the final model decision require concrete-module exports for pair
  helpers, or should they remain internal until E5-F3 integrates execution?
  - Resolved 2026-07-16: keep charged and reduced-property helpers internal to
    `particula.gpu.dynamics.coagulation_funcs`; add no package export in E5.

Classifier diagnostics: none.
