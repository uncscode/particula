# Documentation Updates

## Quick-Start Example

Shipped in this phase: the canonical direct low-level GPU kernel quick-start is
now `docs/Examples/gpu_direct_kernels_quick_start.py`. It is runnable as a
script and shows:

- Availability gate with `WARP_AVAILABLE`.
- Minimal `ParticleData` and `GasData` setup.
- Explicit CPU-to-Warp transfers.
- A direct `condensation_step_gpu` call.
- A direct `coagulation_step_gpu` call.
- Explicit Warp-to-CPU transfer for inspection.
- `device="cpu"` as the default accessible execution target.
- Deferred `particula.gpu.kernels` import until the Warp-enabled execution
  branch.
- Caller-owned coagulation `rng_states` initialized with
  `initialize_rng=True`, `rng_seed=41` for the example call.

## Feature Documentation

Broader guide updates are intentionally separate from this phase, but the
shipped example docstring and output now make the canonical path and explicit
transfer contract discoverable in-repo. Follow-on docs work should update
`docs/Features/data-containers-and-gpu-foundations.md` or an adjacent GPU
feature page to link the quick-start and clarify that direct kernels are
low-level APIs with explicit transfer boundaries.

## Roadmap Documentation

`docs/Features/Roadmap/data-oriented-gpu.md` now records the shipped import-path
decision for phase `E3-F4-P1`: import direct step functions from
`particula.gpu.kernels`, keep top-level `particula.gpu` focused on Warp
availability and transfer helpers, and leave raw helper kernels out of the
documented package-level surface.

## Current Phase Status

- Import-path decision documentation is shipped.
- Focused export regression coverage is shipped and centralized in
  `particula/gpu/tests/kernel_exports_test.py`, including the exact
  `particula.gpu.kernels.__all__` contract, negative package-export checks for
  representative internal helpers, and top-level `particula.gpu`
  non-reexport assertions.
- Duplicate package-export checks were removed from
  `particula/gpu/kernels/tests/coagulation_test.py` as part of the shipped test
  consolidation.
- Runnable quick-start is shipped at
  `docs/Examples/gpu_direct_kernels_quick_start.py`.
- Adjacent example smoke coverage is shipped at
  `particula/gpu/tests/gpu_direct_kernels_example_test.py`, including no-Warp,
  import-deferral, `__main__` / subprocess, Warp CPU, and failure-path checks.
- Broader troubleshooting and cross-document link updates remain for later
  `E3-F4` phases.

## Troubleshooting Content

Add user-facing notes for:

- `WARP_AVAILABLE` is false: install/configure `warp-lang`, or run CPU-only
  non-GPU examples.
- CUDA unavailable: use `device="cpu"`; CUDA snippets are optional.
- Device mismatch: particle, gas, environment, and buffers must be on the same
  Warp device.
- Mixed environment inputs: do not combine scalar `temperature`/`pressure` with
  `environment=`.
- Explicit transfer boundaries: kernels operate on Warp data and do not move
  CPU containers automatically.
