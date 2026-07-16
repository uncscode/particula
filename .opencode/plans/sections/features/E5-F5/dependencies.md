# Dependencies

## Upstream

- E5-F1, Mechanism Configuration and Sampling Contract, must provide stable
  canonical identifiers, capability validation, additive pair-rate/majorant
  dispatch, and the bounded particle-resolved one-pass sampler.
- Shipped E2/E3 GPU foundations provide fp64 `WarpParticleData`, explicit
  environment conversion, fixed-shape buffers, device validation, bounded
  disjoint pair selection, and persistent per-box RNG state.
- `particula/dynamics/coagulation/turbulent_shear_kernel.py` and
  `particula/gas/properties/kinematic_viscosity.py` define approved independent
  CPU references; they are not runtime GPU dependencies.
- Existing GPU dynamic-viscosity and particle-radius helpers provide the device
  primitives needed for ST1956.
- NVIDIA Warp is required for Warp CPU evidence when installed. CUDA is
  optional and must skip cleanly when unavailable.

## Downstream and Siblings

- E5-F6 consumes the turbulent-shear pair-rate and proven majorant for additive
  Brownian/charged/sedimentation/turbulent-shear combinations.
- E5-F7 consumes deterministic pair fixtures plus turbulent-shear-only
  stochastic, conservation, multi-box, buffer, RNG, and device evidence.
- E5-F9 publishes the final support matrix and direct example after combination
  and validation boundaries settle.
- E5-F3 charged execution and E5-F4 sedimentation execution are sibling tracks.
  They may proceed independently after E5-F1 and must extend the same sampler.
- The repository's DNS turbulence modules are explicitly not dependencies and
  must not be imported, ported, or cited as shipped support by E5-F5.

## Phase Ordering

P1 establishes reviewed physics before P2 exposes and validates required box
inputs. P1 and P2 must land before P3 registers executable capability and proves
the majorant/end-to-end state contract. P4 documents only behavior proven by
P1-P3 and remains final. Tests are co-located in every implementation phase;
there is no standalone testing phase. Classifier diagnostics: none.
