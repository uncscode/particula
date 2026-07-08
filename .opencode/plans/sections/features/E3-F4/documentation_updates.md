# Documentation Updates

## Quick-Start Example

Add a direct low-level GPU kernel quick-start under `docs/Examples/`. The
example should be runnable as a script and should show:

- Availability gate with `WARP_AVAILABLE`.
- Minimal `ParticleData` and `GasData` setup.
- Explicit CPU-to-Warp transfers.
- A direct `condensation_step_gpu` call.
- A direct `coagulation_step_gpu` call.
- Explicit Warp-to-CPU transfer for inspection.
- `device="cpu"` as the default accessible execution target.

## Feature Documentation

Update `docs/Features/data-containers-and-gpu-foundations.md` or an adjacent GPU
feature page to link the quick-start and clarify that direct kernels are
low-level APIs with explicit transfer boundaries.

## Roadmap Documentation

Update `docs/Features/Roadmap/data-oriented-gpu.md` to mark the E3-F4 direct-kernel
quick-start as planned or complete once implementation lands.

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
