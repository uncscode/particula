# Data Containers

This example shows the shipped `ParticleData` and `GasData` container schemas,
their single-box shapes, and the optional Warp CPU-backend transfer helpers.

Run the published entrypoint from the repository root:

```bash
python docs/Examples/data_containers_and_gpu_foundations.py
```

The CPU container portion always runs. Warp-backed particle and gas round trips
run only when Warp is available, and they use `device="cpu"` so CUDA is not
required.

- [Feature guide: Data Containers and GPU Foundations](../../Features/data-containers-and-gpu-foundations.md)
- [Runnable entrypoint source on GitHub](https://github.com/Gorkowski/particula/blob/main/docs/Examples/data_containers_and_gpu_foundations.py)
- [Implementation source on GitHub](https://github.com/Gorkowski/particula/blob/main/docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py)
