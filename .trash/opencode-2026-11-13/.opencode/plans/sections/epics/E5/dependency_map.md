# Dependency Map

## Inbound

- Shipped E2 data-model and GPU foundations: particle/gas/environment schemas,
  explicit conversion boundaries, and `WarpParticleData.charge`.
- Shipped E3 kernel correctness and low-level API hardening: fixed-shape
  execution, device validation, persistent RNG state, and bounded Brownian
  sampling behavior.
- Shipped E4 direct-condensation contract and its deferred walkthrough inputs.
- CPU references in charged, sedimentation, turbulent-shear, and combined
  coagulation strategies.
- NVIDIA Warp for required Warp CPU execution when installed; CUDA is optional.

## Outbound

- The next roadmap epic for GPU process completeness and higher-level
  GPU-resident simulation depends on E5's validated mechanism and support
  boundaries.
- User-facing GPU coagulation documentation and direct examples depend on the
  final mechanism configuration and validation matrix.

## Sequencing

- E5-F1 is foundational for E5-F2, E5-F4, and E5-F5.
- E5-F2 must ship before E5-F3.
- E5-F4 and E5-F5 may proceed in parallel after E5-F1.
- E5-F6 depends on E5-F3, E5-F4, and E5-F5.
- E5-F7 depends on all executable mechanism tracks E5-F3 through E5-F6.
- E5-F8 depends only on shipped E4 and may proceed in parallel with E5-F1-F7.
- E5-F9 depends on E5-F6, E5-F7, and E5-F8 and is the final closeout track.
