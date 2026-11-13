# Open Questions

Status: reviewed and answered on 2026-07-10.

## Resolved Decisions

1. The supported public import path is
   `from particula.gpu.kernels import condensation_step_gpu,
   coagulation_step_gpu`.
2. `particula.gpu` remains intentionally non-reexporting for these step
   functions, and `particula/gpu/tests/kernel_exports_test.py` now guards that
   negative contract directly.
3. Raw lower-level symbols such as `apply_coagulation_kernel` stay out of the
   package-level public surface; `particula.gpu.kernels.__all__` is narrowed to
   the two supported step functions only.
4. The canonical direct-kernel quick-start now has its own stable path at
   `docs/Examples/gpu_direct_kernels_quick_start.py`, with smoke coverage in
   `particula/gpu/tests/gpu_direct_kernels_example_test.py`.
5. The shipped quick-start demonstrates the minimum caller-owned coagulation
   state explicitly: allocate `rng_states`, pass them into
   `coagulation_step_gpu(...)`, and initialize them with
   `initialize_rng=True`, `rng_seed=41` for the example call. Repeated-call
   reuse remains documented elsewhere as the broader `E3-F1` contract.
6. Troubleshooting now lives in both the runnable example and the shipped GPU
   docs: `docs/Features/data-containers-and-gpu-foundations.md` and
   `docs/Features/Roadmap/data-oriented-gpu.md`. The example covers immediate
   runnable-path behavior, while the docs explain optional CUDA, Warp-CPU
   fallback, device mismatch, mixed `environment=` plus direct inputs, and
   explicit transfer-boundary / lossy-restore expectations.
