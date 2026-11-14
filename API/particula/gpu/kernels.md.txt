# particula.gpu.kernels

*module*

Expose supported public GPU direct-kernel entry points lazily.

Import direct low-level step functions from this package:

    from particula.gpu.kernels import (
        coagulation_step_gpu,
        activate_slots_gpu,
        condensation_step_gpu,
        dilution_step_gpu,
        nucleation_step_gpu,
        resampling_step_gpu,
        wall_loss_step_gpu,
    )

Lower-level helper kernels and concrete planning records, including
``ResamplingBuffers``, remain importable only from their concrete modules.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/__init__.py#L1-L1)

## particula.gpu.kernels.__getattr__

*function*

```python
def __getattr__(name: str) -> Any
```

Lazily resolve a supported public direct-kernel entry point.

**Parameters**

- `name` (`str`) — Name of the direct GPU kernel entry point to import.

**Returns**

- (`Any`) — The resolved callable entry point.

**Raises**

- `AttributeError` — If ``name`` is not a supported public entry point.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/__init__.py#L45-L64)

**Modules**

- `particula.gpu.kernels._brownian_parallel`
- `particula.gpu.kernels._coagulation_config`
- `particula.gpu.kernels._coagulation_native`
- `particula.gpu.kernels._condensation_native`
- `particula.gpu.kernels._condensation_parallel`
- `particula.gpu.kernels._population_validation`
- `particula.gpu.kernels._wall_loss_native`
- `particula.gpu.kernels.coagulation`
- `particula.gpu.kernels.communication`
- `particula.gpu.kernels.condensation`
- `particula.gpu.kernels.dilution`
- `particula.gpu.kernels.environment`
- `particula.gpu.kernels.exhaustion`
- `particula.gpu.kernels.nucleation`
- `particula.gpu.kernels.slot_management`
- `particula.gpu.kernels.thermodynamics`
- `particula.gpu.kernels.wall_loss`
