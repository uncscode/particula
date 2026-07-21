# Open Questions

- [ ] What exact finite-step update does E6-F1 freeze: exact exponential or a
  bounded explicit substep convention?
  - Resolution gate: E6-F2-P1 must copy the shipped T1 contract and fixtures;
    this plan deliberately does not choose a competing formula.
- [ ] Should `coefficient` and `time_step` both support `(n_boxes,)` Warp arrays,
  or should time remain a finite nonnegative scalar in the first direct API?
  - Recommendation: require per-box coefficient and scalar time unless T1 or
    existing direct-step conventions provide a clear need for per-box time.
- [ ] Should a Python/NumPy floating scalar allocate a private device coefficient
  buffer or launch a scalar kernel parameter?
  - Decision criterion: preserve caller identity and no hidden host-array
    transfer; implementation detail must be tested and documented.
- [ ] Are concentration finiteness/nonnegativity scans required on every call or
  available behind an explicit validation option?
  - Recommendation: validate at the public boundary for atomic guarantees unless
    repository-wide direct-kernel policy establishes a safe explicit opt-out.
- [ ] What `rtol`/`atol` does the T1 fixture matrix justify on Warp CPU and CUDA?
  - Record measured float64 tolerances in P4; do not use undocumented defaults.
