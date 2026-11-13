# particula.gpu.kernels.thermodynamics

*module*

Validate thermodynamic sidecars and refresh GPU vapor pressures in Pa.

This module owns the fixed, device-local thermodynamic sidecar schema and its
concrete-module-only refresh API. The refresh validates caller-owned Warp
buffers before one device launch, then overwrites vapor pressure in Pa from
temperatures in K. Constant models use their first parameter slot; Buck models
use the canonical water/ice equations and reserve all four parameter slots.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/thermodynamics.py#L1-L1)

## particula.gpu.kernels.thermodynamics.THERMODYNAMICS_MODE_BUCK

*attribute* · *module attribute*

```python
THERMODYNAMICS_MODE_BUCK = wp.int32(1)
```

Canonical Buck water/ice mode; all four parameter slots are unused.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/thermodynamics.py#L42-L42)

## particula.gpu.kernels.thermodynamics.THERMODYNAMICS_MODE_CONSTANT

*attribute* · *module attribute*

```python
THERMODYNAMICS_MODE_CONSTANT = wp.int32(0)
```

Constant vapor-pressure mode; parameter zero is pressure in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/thermodynamics.py#L39-L39)

## particula.gpu.kernels.thermodynamics.ThermodynamicsConfig

*class* · *dataclass*

```python
class ThermodynamicsConfig
```

Store caller-owned, device-local thermodynamic model inputs.

The frozen dataclass preserves field bindings, but its caller-owned Warp
buffers remain mutable. This validation sidecar does not itself calculate
vapor pressure or update ``gas.vapor_pressure``. In constant mode,
``parameters[:, 0]`` is vapor pressure in Pa. In Buck mode, all four
parameter slots retain their fixed-schema reservation, but are ignored
because the evaluator uses canonical water/ice equations.

**Attributes**

- `modes` (`Any`) — Per-species ``wp.int32`` model codes with shape ``(n_species,)``.
- `parameters` (`Any`) — Per-species ``wp.float64`` parameters with shape ``(n_species, 4)``. Slot zero is constant-mode pressure in Pa; all four slots are reserved and unused for Buck mode.
- `molar_mass_reference` (`Any`) — Ordered ``wp.float64`` compatibility fingerprint with shape ``(n_species,)``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/thermodynamics.py#L60-L82)

### particula.gpu.kernels.thermodynamics.ThermodynamicsConfig.modes

*attribute* · *instance attribute*

```python
modes: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/thermodynamics.py#L80-L80)

### particula.gpu.kernels.thermodynamics.ThermodynamicsConfig.molar_mass_reference

*attribute* · *instance attribute*

```python
molar_mass_reference: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/thermodynamics.py#L82-L82)

### particula.gpu.kernels.thermodynamics.ThermodynamicsConfig.parameters

*attribute* · *instance attribute*

```python
parameters: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/thermodynamics.py#L81-L81)

### particula.gpu.kernels.thermodynamics.ThermodynamicsConfig.__init__

*method*

```python
def __init__(modes: Any, parameters: Any, molar_mass_reference: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/thermodynamics.py#L0-L0)

## particula.gpu.kernels.thermodynamics.refresh_vapor_pressure_gpu

*function*

```python
def refresh_vapor_pressure_gpu(thermodynamics: ThermodynamicsConfig, gas: Any, temperature: Any) -> None
```

Refresh caller-owned GPU vapor pressure from thermodynamic models.

Import this concrete-module API from
``particula.gpu.kernels.thermodynamics``; it is intentionally not exported
from ``particula.gpu.kernels``. After validating every input, this function
makes one ``(n_boxes, n_species)`` device launch and overwrites every value
in ``gas.vapor_pressure`` in Pa. Invalid calls raise before mutating that
output buffer. Constant models read ``parameters[:, 0]`` in Pa; Buck models
ignore all four reserved parameter slots and use canonical water/ice
equations.

**Parameters**

- `thermodynamics` (`ThermodynamicsConfig`) — Device-local per-species model sidecar.
- `gas` (`Any`) — Caller-owned ``WarpGasData`` whose vapor pressure is overwritten in Pa.
- `temperature` (`Any`) — Device-local ``wp.float64`` temperatures in K with shape ``(n_boxes,)``.

**Returns**

- (`None`) — ``None``. The function mutates only ``gas.vapor_pressure``.

**Raises**

- `ValueError` — If an input violates the device-array contract or the
configuration is invalid. ``gas.vapor_pressure`` is unchanged.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/thermodynamics.py#L348-L407)

## particula.gpu.kernels.thermodynamics.validate_thermodynamics_config

*function*

```python
def validate_thermodynamics_config(thermodynamics: ThermodynamicsConfig | Any | None, n_species: int, device: Any, gas_molar_mass: Any, caller_name: str) -> ThermodynamicsConfig
```

Validate a thermodynamic sidecar and return its identical object.

Structural metadata is validated before any device-to-host readback. Valid
caller buffers are never allocated, replaced, or mutated. This
validation-only function does not launch kernels, evaluate thermodynamic
parameters, calculate vapor pressure, or refresh ``gas.vapor_pressure``.

**Parameters**

- `thermodynamics` (`ThermodynamicsConfig | Any | None`) — Candidate caller-owned configuration.
- `n_species` (`int`) — Expected species count.
- `device` (`Any`) — Active particle and gas Warp device.
- `gas_molar_mass` (`Any`) — Device-local gas molar masses used as the ordered compatibility fingerprint.
- `caller_name` (`str`) — Public entry point used in error messages.

**Returns**

- (`ThermodynamicsConfig`) — The exact ``ThermodynamicsConfig`` instance supplied by the caller.

**Raises**

- `ValueError` — If the configuration schema, device locality, values, or
ordered molar-mass fingerprint is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/thermodynamics.py#L140-L245)
