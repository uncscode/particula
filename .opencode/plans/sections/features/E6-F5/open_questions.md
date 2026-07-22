# Open Questions

All E6-F5 planning questions were resolved on 2026-07-21 from fixed-slot and
caller-owned-buffer conventions.

- [x] What makes a slot available for overwrite?
  - Decision: only the canonical all-zero mass, concentration, and charge
    record. Contradictory partial state is invalid.
- [x] How are multiple requests assigned?
  - Decision: valid request rank maps to ascending free-slot index independently
    in each box.
- [x] Does E6-F5 recover from insufficient capacity?
  - Decision: no. It fails before mutation; E6-F6 owns recovery policies.
- [x] Where do diagnostics live?
  - Decision: in result values or optional caller-owned sidecars, never as new
    particle-container fields.
- [x] How are zero-box and zero-slot containers handled?
  - Decision: reject `n_boxes == 0` on CPU and Warp. Accept zero slots as an
    exact no-op only when every request count is zero; any positive request is a
    capacity error before writes.
- [x] Which diagnostic-buffer aliases are accepted?
  - Decision: none. Reject every writable overlap with particle state, request
    arrays, scratch, or another diagnostic before clearing outputs. Dtype
    separation is not a sufficient alias guarantee.
