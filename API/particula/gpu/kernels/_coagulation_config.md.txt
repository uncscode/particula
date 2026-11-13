# particula.gpu.kernels._coagulation_config

*module*

Host-only configuration resolution for GPU coagulation mechanisms.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L1-L1)

## particula.gpu.kernels._coagulation_config.BROWNIAN_MECHANISM

*attribute* · *module attribute*

```python
BROWNIAN_MECHANISM = 'brownian'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L6-L6)

## particula.gpu.kernels._coagulation_config.BROWNIAN_MECHANISM_FLAG

*attribute* · *module attribute*

```python
BROWNIAN_MECHANISM_FLAG = 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L18-L18)

## particula.gpu.kernels._coagulation_config.CANONICAL_COAGULATION_MECHANISMS

*attribute* · *module attribute*

```python
CANONICAL_COAGULATION_MECHANISMS = (BROWNIAN_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM, SEDIMENTATION_SP2016_MECHANISM, TURBULENT_SHEAR_ST1956_MECHANISM)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L11-L16)

## particula.gpu.kernels._coagulation_config.CHARGED_HARD_SPHERE_MECHANISM

*attribute* · *module attribute*

```python
CHARGED_HARD_SPHERE_MECHANISM = 'charged_hard_sphere'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L7-L7)

## particula.gpu.kernels._coagulation_config.CHARGED_HARD_SPHERE_MECHANISM_FLAG

*attribute* · *module attribute*

```python
CHARGED_HARD_SPHERE_MECHANISM_FLAG = 2
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L19-L19)

## particula.gpu.kernels._coagulation_config.SEDIMENTATION_SP2016_MECHANISM

*attribute* · *module attribute*

```python
SEDIMENTATION_SP2016_MECHANISM = 'sedimentation_sp2016'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L8-L8)

## particula.gpu.kernels._coagulation_config.SEDIMENTATION_SP2016_MECHANISM_FLAG

*attribute* · *module attribute*

```python
SEDIMENTATION_SP2016_MECHANISM_FLAG = 4
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L20-L20)

## particula.gpu.kernels._coagulation_config.TURBULENT_SHEAR_ST1956_MECHANISM

*attribute* · *module attribute*

```python
TURBULENT_SHEAR_ST1956_MECHANISM = 'turbulent_shear_st1956'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L9-L9)

## particula.gpu.kernels._coagulation_config.TURBULENT_SHEAR_ST1956_MECHANISM_FLAG

*attribute* · *module attribute*

```python
TURBULENT_SHEAR_ST1956_MECHANISM_FLAG = 8
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L21-L21)

## particula.gpu.kernels._coagulation_config._COAGULATION_MECHANISM_FLAGS

*attribute* · *module attribute*

```python
_COAGULATION_MECHANISM_FLAGS = MappingProxyType({BROWNIAN_MECHANISM: BROWNIAN_MECHANISM_FLAG, CHARGED_HARD_SPHERE_MECHANISM: CHARGED_HARD_SPHERE_MECHANISM_FLAG, SEDIMENTATION_SP2016_MECHANISM: SEDIMENTATION_SP2016_MECHANISM_FLAG, TURBULENT_SHEAR_ST1956_MECHANISM: TURBULENT_SHEAR_ST1956_MECHANISM_FLAG})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L23-L30)

## particula.gpu.kernels._coagulation_config.CoagulationMechanismConfig

*class* · *dataclass*

```python
class CoagulationMechanismConfig
```

Configure host-side coagulation mechanism selection.

**Attributes**

- `mechanisms` (`tuple[str, ...] | None`) — Requested identifiers, or ``None`` for Brownian.
- `distribution_type` (`str`) — Required representation; only ``"particle_resolved"`` is structurally supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L33-L44)

### particula.gpu.kernels._coagulation_config.CoagulationMechanismConfig.distribution_type

*attribute* · *class attribute* · *instance attribute*

```python
distribution_type: str = 'particle_resolved'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L44-L44)

### particula.gpu.kernels._coagulation_config.CoagulationMechanismConfig.mechanisms

*attribute* · *class attribute* · *instance attribute*

```python
mechanisms: tuple[str, ...] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L43-L43)

### particula.gpu.kernels._coagulation_config.CoagulationMechanismConfig.__init__

*method*

```python
def __init__(mechanisms: tuple[str, ...] | None = None, distribution_type: str = 'particle_resolved') -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L0-L0)

## particula.gpu.kernels._coagulation_config._ResolvedCoagulationMechanismConfig

*class* · *dataclass*

```python
class _ResolvedCoagulationMechanismConfig
```

Store a normalized, structurally valid mechanism configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L47-L53)

### particula.gpu.kernels._coagulation_config._ResolvedCoagulationMechanismConfig.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L52-L52)

### particula.gpu.kernels._coagulation_config._ResolvedCoagulationMechanismConfig.mask

*attribute* · *instance attribute*

```python
mask: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L53-L53)

### particula.gpu.kernels._coagulation_config._ResolvedCoagulationMechanismConfig.mechanisms

*attribute* · *instance attribute*

```python
mechanisms: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L51-L51)

### particula.gpu.kernels._coagulation_config._ResolvedCoagulationMechanismConfig.__init__

*method*

```python
def __init__(mechanisms: tuple[str, ...], distribution_type: str, mask: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L0-L0)

## particula.gpu.kernels._coagulation_config.resolve_coagulation_mechanism_config

*function*

```python
def resolve_coagulation_mechanism_config(config: CoagulationMechanismConfig) -> _ResolvedCoagulationMechanismConfig
```

Resolve a configuration through host-side structural validation.

**Parameters**

- `config` (`CoagulationMechanismConfig`) — Configuration whose distribution type and mechanism identifiers are validated and normalized into canonical mechanism order.

**Returns**

- (`_ResolvedCoagulationMechanismConfig`) — The normalized configuration, including its additive mechanism mask.

**Raises**

- `ValueError` — If the distribution type is unsupported, mechanisms are not
a non-empty tuple of strings, a mechanism is duplicated or unknown.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L56-L109)

## particula.gpu.kernels._coagulation_config.validate_coagulation_mechanism_capabilities

*function*

```python
def validate_coagulation_mechanism_capabilities(resolved: _ResolvedCoagulationMechanismConfig) -> None
```

Enforce the executable and deferred mechanism-mask boundary.

**Parameters**

- `resolved` (`_ResolvedCoagulationMechanismConfig`) — Structurally valid, normalized configuration to check.

**Returns**

- (`None`) — None.

**Raises**

- `ValueError` — If the mechanism mask is a deferred three-way combination.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L112-L128)
