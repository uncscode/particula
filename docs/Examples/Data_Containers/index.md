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

For the low-level direct condensation path, use the separate canonical
quick-start:

```bash
python docs/Examples/gpu_direct_kernels_quick_start.py
```

That runnable script is the direct GPU condensation example. It
demonstrates explicit `to_warp_*` / `from_warp_*` transfers, lazy imports from
`particula.gpu.kernels`, two fixed-four-substep condensation calls, and
caller-owned fp64 scratch, physical-property, latent-heat, and energy sidecars
reused on Warp `device="cpu"` by default. It does not invoke coagulation or
configure RNG state.

For the bounded, low-level particle-resolved Brownian coagulation path, run:

```bash
python docs/Examples/gpu_coagulation_direct.py
```

This standalone example explicitly transfers `ParticleData` to Warp, defaults
to Warp `device="cpu"`, and makes two supported Brownian direct calls. Its
collision-pair, collision-count, and persistent RNG-state sidecars are
caller-owned and reused across those calls before an explicit CPU checkpoint
restore. Warp imports and all conversion/kernel work are skipped when Warp is
unavailable or `PARTICULA_EXAMPLE_FORCE_NO_WARP=1`; this disabled route has no
CPU coagulation fallback. The example is a direct-kernel path, not a Runnable
API.

- [Feature guide: Data Containers and GPU Foundations](../../Features/data-containers-and-gpu-foundations.md)
- [Runnable entrypoint source on GitHub](https://github.com/Gorkowski/particula/blob/main/docs/Examples/data_containers_and_gpu_foundations.py)
- [Direct GPU kernels quick-start source on GitHub](https://github.com/Gorkowski/particula/blob/main/docs/Examples/gpu_direct_kernels_quick_start.py)
- [Direct GPU coagulation source on GitHub](https://github.com/Gorkowski/particula/blob/main/docs/Examples/gpu_coagulation_direct.py)
