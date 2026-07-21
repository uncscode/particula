# Infrastructure Reuse

- `ParticleData` shape and ownership rules in
  `particula/particles/particle_data.py:47-185` provide `(boxes, slots, species)`
  masses, `(boxes, slots)` concentration/charge, and derived total mass.
- `WarpParticleData` in `particula/gpu/warp_types.py:24-78` mirrors those arrays
  as fixed-shape `wp.float64` device storage; do not add active-count fields.
- Existing coagulation validation at
  `particula/gpu/kernels/coagulation.py:784-816` already treats selector-eligible
  slots as positive concentration plus positive total volume and demonstrates
  read-only status/count kernels before mutation.
- Sedimentation validation in
  `particula/gpu/kernels/coagulation.py:875-927` demonstrates box-local scanning,
  finite/nonnegative checks, active counting, and status-code precedence.
- Compact active-index and `-1` sentinel patterns in
  `particula/gpu/kernels/coagulation.py:359-719` should guide deterministic
  fixed-shape index sidecars, without importing private coagulation helpers.
- Conversion code in `particula/gpu/conversion.py` is the explicit CPU/Warp
  boundary; activation must not call it internally.
- Existing direct condensation/coagulation entry points demonstrate same-device
  validation, caller-owned output identity, and preflight-before-launch order.
- `particula/gpu/tests/cuda_availability.py` supplies mandatory Warp CPU and
  optional CUDA test parametrization.
- Parent E6 roadmap requirements are recorded in
  `docs/Features/Roadmap/data-oriented-gpu.md:1219-1237`.
