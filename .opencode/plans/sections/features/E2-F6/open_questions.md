# E2-F6 Open Questions

- What tolerance thresholds should define acceptable small-particle mass and
  radius fidelity for NPF-to-droplet coexistence cases?
- Should the final report live in `docs/Features/` or next to roadmap material
  under `docs/Features/Roadmap/`?
- Which mixed-precision candidate is most relevant for near-term GPU work:
  `fp32` storage with `fp64` accumulation, `fp64` storage with `fp32` transient
  buffers, or a different split?
- How should gas-mass conservation be represented if E2-F2/E2-F3 environment
  containers are still evolving during implementation?
- Should log-mass or reference-mass scaling be treated as a quantitative
  candidate in this feature, or only documented as future work if initial
  evidence shows absolute `fp64` is insufficient?
