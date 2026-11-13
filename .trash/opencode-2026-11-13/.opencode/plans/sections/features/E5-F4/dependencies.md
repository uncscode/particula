# Dependencies

## Upstream

- E5-F1, Mechanism Configuration and Sampling Contract, must provide stable
  canonical identifiers, the capability matrix, additive pair-rate/majorant
  dispatch, and the bounded one-pass particle-resolved sampler.
- Shipped E2/E3 GPU foundations provide fixed-shape fp64 `WarpParticleData`,
  explicit environment conversion, device validation, collision buffers,
  bounded disjoint pair sampling, and persistent per-box RNG state.
- The CPU references in
  `particula/dynamics/coagulation/sedimentation_kernel.py` and
  `particula/particles/properties/settling_velocity.py` define the approved
  SP2016 unit-efficiency and Stokes/slip equations. They are independent test
  references, not runtime GPU dependencies.
- Existing GPU gas and particle property helpers provide dynamic viscosity,
  molecular mean free path, Knudsen number, slip correction, and radius.
- NVIDIA Warp is required for Warp CPU evidence when installed. CUDA is
  optional and must skip cleanly when unavailable.

## Downstream and Siblings

- E5-F6 consumes the sedimentation pair-rate and proven majorant branches for
  Brownian-plus-sedimentation and broader additive combinations.
- E5-F7 consumes deterministic pair/property fixtures and sedimentation-only
  stochastic, conservation, multi-box, buffer, RNG, and device evidence.
- E5-F9 documents the final support matrix and direct example after E5-F6/F7
  settle the complete combination boundary.
- E5-F3 charged execution and E5-F5 turbulent-shear execution are sibling
  mechanism tracks. They may proceed independently after E5-F1 and must extend,
  not fork, the shared sampler.

## Phase Ordering

P1 establishes reviewed property and pair physics before P2 can prove the
majorant and dispatch it. P2 must land before P3 registers public execution and
state-safety evidence. P4 documents only behavior proven by P1-P3 and remains
final. Every implementation phase includes co-located tests; there is no
standalone testing phase. Classifier diagnostics: none.

Issue #1350 is a completed documentation-only follow-up to #1349. It records
existing E5-F4 evidence and boundaries; it does not deliver E5-F6 additive
combinations, E5-F7 evidence consumption, or E5-F9's final matrix/example.
