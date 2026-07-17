# Documentation Updates

- [x] Update internal module, dispatcher, and sampler documentation in
  `particula/gpu/kernels/coagulation.py` with the private exact-mask boundary,
  shared scheduler/RNG path, and cleared settling-velocity scratch ownership.
- User-facing sedimentation documentation remains deferred to P4: P2
  sedimentation execution is private and test-only, and public
  `coagulation_step_gpu` configuration still rejects it.
- Document the new scalar helpers in
  `particula/gpu/dynamics/coagulation_funcs.py` with the SP2016 and Stokes/slip
  equations, units, and citations.
- Update `docs/Features/data-containers-and-gpu-foundations.md` with the direct
  sedimentation support row, temporary property ownership, fixed-shape fp64
  boundary, caller-owned buffers/RNG, and supported devices.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` with E5-F4 delivery state,
  evidence commands, downstream E5-F6/F7 dependencies, and deferred variants.
- Cross-reference E5-F9 for the final user-facing direct coagulation example and
  consolidated support matrix; E5-F4 should not prematurely claim additive
  combinations.
- Update E5 and E5-F4 plan sections as phases ship, including issue numbers,
  statuses, validation evidence, and resolved questions.

Required support wording: particle-resolved direct low-level execution only;
SP2016 geometric sedimentation; Stokes settling with Cunningham slip correction;
collision efficiency fixed at 1; Warp CPU required when installed; CUDA
optional; no non-unit efficiency, drag/DNS variant, high-level runnable, CPU
fallback, hidden transfer, or additive combination before E5-F6.
