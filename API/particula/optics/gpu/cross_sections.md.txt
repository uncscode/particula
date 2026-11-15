# particula.optics.gpu.cross_sections

*module*

Per-particle dimensional optical cross sections with explicit status.

Cross sections are Q*pi*r². Efficiencies follow Bohren and Huffman (1983),
https://doi.org/10.1002/9783527618156. Particle absorption remains separate
from any later energy transfer or aerosol/environment state mutation.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/cross_sections.py#L1-L1)

## particula.optics.gpu.cross_sections._PI

*attribute* · *module attribute*

```python
_PI = wp.constant(wp.float64(math.pi))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/cross_sections.py#L14-L14)

## particula.optics.gpu.cross_sections.cross_sections_kernel

*function*

```python
def cross_sections_kernel(efficiencies: wp.array2d(dtype=wp.float64), radius: wp.array(dtype=wp.float64), input_status: wp.array(dtype=wp.int32), cross_sections: wp.array2d(dtype=wp.float64), status: wp.array(dtype=wp.int32))
```

Convert (cases,7) efficiencies to (cases,5) cross sections [m²].

**Parameters**

- `efficiencies` (`wp.array2d(dtype=wp.float64)`) — Standard Mie order: ext, sca, abs, g, back, pr, valid.
- `radius` (`wp.array(dtype=wp.float64)`) — Nonnegative finite SI radii [m], shape (cases,).
- `input_status` (`wp.array(dtype=wp.int32)`) — Upstream coefficient/efficiency status, shape (cases,).
- `cross_sections` (`wp.array2d(dtype=wp.float64)`) — Caller-owned output (cases,5): ext, sca, abs, back, pr.
- `status` (`wp.array(dtype=wp.int32)`) — Caller-owned (cases,) int32: 0 success, 1 invalid input, 2 upstream failure, 3 nonfinite result.

Launch one thread per case, with all arrays on the same device. Shapes
and float64/int32 dtypes are the caller's responsibility. Observe status
before using outputs; a failed case can contain partial results. No
synchronization, transfer, allocation or source mutation occurs here.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/cross_sections.py#L17-L64)
