# Architecture Design

## Design summary

`WarpEnvironmentData` is a thin Warp struct that mirrors the CPU
`EnvironmentData` schema from `E2-F2`. It stores only numeric per-box state on
the selected Warp device. Conversion helpers are the only sanctioned boundary
between CPU NumPy arrays and Warp arrays.

## Proposed API

```python
@wp.struct
class WarpEnvironmentData:
    temperature: wp.array(dtype=wp.float64)
    pressure: wp.array(dtype=wp.float64)
    # Additional E2-F2 numeric environment fields, each shaped (n_boxes,).
```

```python
def to_warp_environment_data(
    data: EnvironmentData,
    device: str = "cuda",
    copy: bool = True,
) -> WarpEnvironmentData: ...

def from_warp_environment_data(
    gpu_data: WarpEnvironmentData,
    sync: bool = True,
) -> EnvironmentData: ...
```

## Data flow

1. CPU simulation setup creates or receives `EnvironmentData`.
2. Callers explicitly invoke `to_warp_environment_data` before GPU work.
3. Kernels receive `WarpEnvironmentData` or individual fields in later tracks.
4. Callers explicitly invoke `from_warp_environment_data` when CPU state is
   required again.

## Boundary principles

- No conversion helper should be called implicitly by kernels, runnables, or
  strategy objects.
- The CPU helper controls copy semantics; the Warp-to-CPU helper controls
  synchronization semantics.
- Device validation remains centralized in `particula/gpu/conversion.py`.
- Existing scalar kernel APIs remain stable until later migration tracks.

## Compatibility

The API follows current `ParticleData` and `GasData` Warp patterns, so users who
already transfer those containers see the same parameters, exceptions, and test
behavior.
