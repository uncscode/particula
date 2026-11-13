# particula.gpu

*module*

Experimental GPU acceleration module for particula.

This module provides GPU-accelerated data containers and CPU↔Warp
conversion helpers using NVIDIA Warp for high-performance particle
simulations.

Its low-level containers, conversion helpers, and explicit-transfer/direct-
kernel workflow are experimental. Their current import paths and caller-owned
transfer model remain supported.

The GPU module enables:
- GPU-resident particle, gas, and environment data via Warp structs
- Efficient multi-box CFD simulations with batch dimensions
- Manual transfer control between CPU and GPU (to_warp/from_warp)
- Top-level availability and transfer helpers, while direct GPU step imports
  live under ``particula.gpu.kernels``

> **Example**
>
> >>> import warp as wp
> >>> from particula.gpu import WarpParticleData, WarpGasData
> >>> wp.init()
> >>> # Create particle data for 2 boxes, 100 particles, 3 species
> >>> data = WarpParticleData()
> >>> data.masses = wp.zeros((2, 100, 3), dtype=wp.float64)

> **References**
>
> NVIDIA Warp documentation: https://nvidia.github.io/warp/

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/__init__.py#L1-L1)

## particula.gpu.__getattr__

*function*

```python
def __getattr__(name: str) -> Any
```

Lazily resolve optional Warp-backed support and transfer symbols.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/__init__.py#L88-L122)

**Modules**

- `particula.gpu.conversion`
- `particula.gpu.dynamics`
- `particula.gpu.kernels`
- `particula.gpu.nucleation_species_mapping`
- `particula.gpu.population_metadata`
- `particula.gpu.properties`
- `particula.gpu.species_mapping`
- `particula.gpu.warp_types`
