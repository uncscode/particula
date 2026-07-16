# Open Questions

- [ ] Which CPU charged variants receive stable first-release identifiers?
  - Owner: E5-F1/E5-F2.
  - Constraint: excluded variants must remain explicit and unavailable; do not
    use a generic `charged` name if it hides materially different formulas.
- [ ] Should `CoagulationMechanismConfig` be exported from
  `particula.gpu.kernels`, or remain a concrete-module-only sidecar until E5-F6
  stabilizes all fields?
  - Owner: E5-F1-P1.
  - Proposed default: concrete-module-only, matching specialized low-level
    sidecars, while `coagulation_step_gpu` remains the exported step.
- [ ] Should user ordering be rejected unless canonical or accepted and
  normalized to canonical order?
  - Owner: E5-F1-P1.
  - Proposed resolution: accept any unique ordering and normalize it so
    equivalent sets produce the same mask and launch behavior.
- [ ] How should later scalar/per-box dissipation and fluid-density inputs be
  grouped without coupling unrelated mechanisms?
  - Owner: E5-F1/E5-F5.
  - Constraint: arrays remain explicit, active-device, shape `(n_boxes,)`, and
    validate before mutation; F1 should reserve extension points, not add
    unused arrays.
- [ ] Should invalid device-computed rate/majorant values set a diagnostic
  buffer or simply suppress acceptance after host preflight?
  - Owner: E5-F1-P2/E5-F6.
  - Constraint: no hidden host synchronization or public debug-only API.

Classifier diagnostics: none.
