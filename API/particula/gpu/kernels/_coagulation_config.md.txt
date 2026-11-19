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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L22-L22)

## particula.gpu.kernels._coagulation_config.CANONICAL_COAGULATION_MECHANISMS

*attribute* · *module attribute*

```python
CANONICAL_COAGULATION_MECHANISMS = (BROWNIAN_MECHANISM, VAN_DER_WAALS_MECHANISM, CHARGED_HARD_SPHERE_MECHANISM, SEDIMENTATION_SP2016_MECHANISM, TURBULENT_SHEAR_ST1956_MECHANISM)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L14-L20)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L23-L23)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L24-L24)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L25-L25)

## particula.gpu.kernels._coagulation_config.VAN_DER_WAALS_MECHANISM

*attribute* · *module attribute*

```python
VAN_DER_WAALS_MECHANISM = 'van_der_waals'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L11-L11)

## particula.gpu.kernels._coagulation_config.VAN_DER_WAALS_MECHANISM_FLAG

*attribute* · *module attribute*

```python
VAN_DER_WAALS_MECHANISM_FLAG = 16
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L12-L12)

## particula.gpu.kernels._coagulation_config._COAGULATION_MECHANISM_FLAGS

*attribute* · *module attribute*

```python
_COAGULATION_MECHANISM_FLAGS = MappingProxyType({VAN_DER_WAALS_MECHANISM: VAN_DER_WAALS_MECHANISM_FLAG, BROWNIAN_MECHANISM: BROWNIAN_MECHANISM_FLAG, CHARGED_HARD_SPHERE_MECHANISM: CHARGED_HARD_SPHERE_MECHANISM_FLAG, SEDIMENTATION_SP2016_MECHANISM: SEDIMENTATION_SP2016_MECHANISM_FLAG, TURBULENT_SHEAR_ST1956_MECHANISM: TURBULENT_SHEAR_ST1956_MECHANISM_FLAG})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L27-L35)

## particula.gpu.kernels._coagulation_config.CoagulationMechanismConfig

*class* · *dataclass*

```python
class CoagulationMechanismConfig
```

Configure host-side coagulation mechanism selection.

**Attributes**

- `mechanisms` (`tuple[str, ...] | None`) — Requested identifiers, or ``None`` for Brownian.
- `distribution_type` (`str`) — Required representation; only ``"particle_resolved"`` is structurally supported.
- `charged_model` (`str`) — Alternative charged Brownian fit: hard_sphere, dyachkov2007, gatti2008, gopalakrishnan2012 or chahl2019. Nondefault selection requires the charged mechanism.
- `hamaker_constant` (`float | None`) — Effective dispersion constant across air [J].
- `relative_permittivity` (`float | None`) — Static relative permittivity of both spheres.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L38-L57)

### particula.gpu.kernels._coagulation_config.CoagulationMechanismConfig.charged_model

*attribute* · *class attribute* · *instance attribute*

```python
charged_model: str = 'hard_sphere'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L55-L55)

### particula.gpu.kernels._coagulation_config.CoagulationMechanismConfig.distribution_type

*attribute* · *class attribute* · *instance attribute*

```python
distribution_type: str = 'particle_resolved'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L54-L54)

### particula.gpu.kernels._coagulation_config.CoagulationMechanismConfig.hamaker_constant

*attribute* · *class attribute* · *instance attribute*

```python
hamaker_constant: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L56-L56)

### particula.gpu.kernels._coagulation_config.CoagulationMechanismConfig.mechanisms

*attribute* · *class attribute* · *instance attribute*

```python
mechanisms: tuple[str, ...] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L53-L53)

### particula.gpu.kernels._coagulation_config.CoagulationMechanismConfig.relative_permittivity

*attribute* · *class attribute* · *instance attribute*

```python
relative_permittivity: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L57-L57)

### particula.gpu.kernels._coagulation_config.CoagulationMechanismConfig.__init__

*method*

```python
def __init__(mechanisms: tuple[str, ...] | None = None, distribution_type: str = 'particle_resolved', charged_model: str = 'hard_sphere', hamaker_constant: float | None = None, relative_permittivity: float | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L0-L0)

## particula.gpu.kernels._coagulation_config._ResolvedCoagulationMechanismConfig

*class* · *dataclass*

```python
class _ResolvedCoagulationMechanismConfig
```

Store a normalized, structurally valid mechanism configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L60-L67)

### particula.gpu.kernels._coagulation_config._ResolvedCoagulationMechanismConfig.charged_model

*attribute* · *class attribute* · *instance attribute*

```python
charged_model: int = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L67-L67)

### particula.gpu.kernels._coagulation_config._ResolvedCoagulationMechanismConfig.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L65-L65)

### particula.gpu.kernels._coagulation_config._ResolvedCoagulationMechanismConfig.mask

*attribute* · *instance attribute*

```python
mask: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L66-L66)

### particula.gpu.kernels._coagulation_config._ResolvedCoagulationMechanismConfig.mechanisms

*attribute* · *instance attribute*

```python
mechanisms: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L64-L64)

### particula.gpu.kernels._coagulation_config._ResolvedCoagulationMechanismConfig.__init__

*method*

```python
def __init__(mechanisms: tuple[str, ...], distribution_type: str, mask: int, charged_model: int = 0) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L0-L0)

## particula.gpu.kernels._coagulation_config._validate_dispersion_config

*function*

```python
def _validate_dispersion_config(config, seen)
```

Check explicit material inputs and overlapping thermal mechanisms.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L70-L86)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L89-L158)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_config.py#L161-L177)
