# Documentation Updates

- [x] Update internal module, dispatcher, and sampler documentation in
  `particula/gpu/kernels/coagulation.py` with the public exact-mask boundary,
  shared scheduler/RNG path, preflight ordering, and scratch ownership.
- [x] Update `docs/Features/data-containers-and-gpu-foundations.md` with the
  canonical `CoagulationMechanismConfig(("sedimentation_sp2016",))` direct
  invocation, particle-resolved fp64/unit-efficiency boundary, caller-owned
  output and persistent RNG contracts, no-transfer/no-fallback guarantee, and
  deferred combinations and variants.
- [x] Document the scalar helpers in
  `particula/gpu/dynamics/coagulation_funcs.py` with the SP2016 and Stokes/slip
  equations, units, and citations.
- [x] Document the direct sedimentation support row, temporary property
  ownership, fixed-shape fp64 boundary, caller-owned buffers/RNG, and supported
  devices in the user-facing data-container guide.
- [x] Keep support wording explicit that additive combinations are deferred.
- [x] Update E5-F4 plan sections with Issue #1349 delivery evidence.
- [x] Complete Issue #1350 documentation follow-up: add the user-facing SP2016
  equation, m³/s units, citation, fixed-unit-efficiency scope, focused
  sedimentation evidence wording, and confirmed input/ownership/preflight/device
  boundaries without replacing the #1349 P3/P4 delivery record.

Required support wording: particle-resolved direct low-level execution only;
SP2016 geometric sedimentation; Stokes settling with Cunningham slip correction;
collision efficiency fixed at 1; Warp CPU required when installed; CUDA
optional; no non-unit efficiency, drag/DNS variant, high-level runnable, CPU
fallback, hidden transfer, or additive combination before E5-F6.

Issue #1350 preserves downstream ownership: E5-F6 owns additive combinations,
E5-F7 consumes sedimentation evidence, and E5-F9 owns the final consolidated
support matrix and direct example.
