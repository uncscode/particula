# Infrastructure Reuse

- `particula/gpu/kernels/__init__.py:1-35` — preserve the canonical lazy public
  import `from particula.gpu.kernels import coagulation_step_gpu`; do not expose
  the step from top-level `particula.gpu`.
- `particula/gpu/kernels/coagulation.py:792-1022` — document the authoritative
  direct-step signature, in-place particle mutation, collision outputs,
  environment/volume normalization, preflight behavior, and RNG ownership as
  extended by E5-F1 through E5-F6.
- `particula/gpu/conversion.py` and `particula/gpu/warp_types.py` — reuse public
  explicit particle transfer helpers and fixed-shape `WarpParticleData`; do not
  introduce hidden transfers or new container ownership.
- `docs/Examples/gpu_direct_kernels_quick_start.py:103-139,172-341` — follow
  the lazy Warp import, Warp CPU default, deterministic metadata, explicit
  restore, and clean no-Warp branch patterns for the coagulation example.
- `particula/gpu/tests/gpu_direct_kernels_example_test.py` — mirror the existing
  example import/no-Warp/runtime test structure in a dedicated
  `gpu_coagulation_direct_example_test.py`.
- `particula/gpu/kernels/tests/coagulation_test.py` — cite, rather than
  duplicate, E5-F7's deterministic parity, aggregate stochastic, conservation,
  inactive-slot, multi-box, buffer, and persistent-RNG evidence.
- `.opencode/guides/testing_guide.md:166-208` — reuse Warp CPU required-when-
  installed, CUDA optional, marker, warning, and focused release-command policy.
- `docs/Features/data-containers-and-gpu-foundations.md:454-525` — refine the
  existing support-boundary table and user guidance without broadening it.
- `docs/Features/Roadmap/data-oriented-gpu.md:989-1034` and
  `docs/Features/Roadmap/index.md:61-128` — replace active/placeholder E5 text
  with dependency-verified IDs, artifacts, and status transitions.
- `.opencode/plans/sections/epics/E5/dependency_map.md:23-31` and
  `success_metrics.md:3-32` — use the parent dependency order and exit checks as
  the closeout gate; sibling feature plans remain the authoritative phase and
  artifact handoffs.
