# Open Questions

- [ ] Which exact canonical per-field `rtol`/`atol` pairs from
  `particula/gpu/kernels/tests/condensation_test.py` apply to the selected
  walkthrough fixture?
  - Resolve in P1 before fixture acceptance. Do not reuse conservation or
    energy tolerances as physics tolerances and do not relax shipped bounds.
- [ ] Should the optional CUDA run be exposed by a command-line flag or remain a
  pytest-only selection?
  - Default remains Warp CPU. Resolve in P1 based on existing example/device
    conventions; either choice must skip cleanly without CUDA.
- [ ] Does Epic F formally accept both `thermal_work` consumption/temperature
  feedback and adaptive condensation stepping, or should either receive a
  separately approved numerical-method feature before E5 closes?
  - Resolve with the E5/E6 roadmap owner in P3. Until accepted, the ownership
    table must mark the gate as pending rather than imply support.
- [ ] What plan ID will own phase-aware surface tension and BAT activity?
  - Open by design: P3 records the approved condensation-physics expansion lane
    and requires a concrete child plan ID before implementation begins.
- [x] Is CUDA required for E5-F8 completion?
  - Resolved 2026-07-15: No. Warp CPU is required whenever Warp is installed;
    CUDA is optional additive evidence and must skip cleanly when unavailable.
- [x] Does this walkthrough establish high-level CPU strategy parity?
  - Resolved 2026-07-15: No. It establishes only the bounded independent NumPy
    fixed-four-substep oracle versus low-level Warp direct-kernel criteria.
