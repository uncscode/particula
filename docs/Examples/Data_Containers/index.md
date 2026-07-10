# Data Containers

This guide explains the shipped `ParticleData` and `GasData` container
schemas, their single-box shapes, and the optional Warp CPU-backend transfer
helpers.

Run the published entrypoint from the repository root:

```bash
python docs/Examples/data_containers_and_gpu_foundations.py
```

The top-level script above is the canonical runnable example. This page is
supporting context only.

The CPU container portion always runs. Warp-backed particle and gas round
trips run only when Warp is available, and they use `device="cpu"` so CUDA is
not required.

For the low-level direct kernel path, use the separate canonical quick-start:

```bash
python docs/Examples/gpu_direct_kernels_quick_start.py
```

That runnable script is the single direct GPU kernel example. It demonstrates
explicit `to_warp_*` / `from_warp_*` transfers, lazy imports from
`particula.gpu.kernels`, one condensation kernel call, one coagulation kernel
call, caller-owned `rng_states`, and Warp `device="cpu"` by default.

- [Feature guide: Data Containers and GPU Foundations](../../Features/data-containers-and-gpu-foundations.md)
- [Runnable entrypoint source on GitHub](https://github.com/Gorkowski/particula/blob/main/docs/Examples/data_containers_and_gpu_foundations.py)
- [Direct GPU kernels quick-start source on GitHub](https://github.com/Gorkowski/particula/blob/main/docs/Examples/gpu_direct_kernels_quick_start.py)
