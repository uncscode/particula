# particula.aerosol_builder

*module*

Fluent assembly of an identity-preserving native Aerosol.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_builder.py#L1-L1)

## particula.aerosol_builder.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_builder.py#L11-L11)

## particula.aerosol_builder.AerosolBuilder

*class*

```python
class AerosolBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Retain three supplied CPU containers and validate them at build time.

No default containers, copies, species filtering, or physics configuration
are created. Repeated builds share the supplied data by identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_builder.py#L14-L72)

### particula.aerosol_builder.AerosolBuilder.environment

*attribute* · *instance attribute*

```python
environment: EnvironmentData | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_builder.py#L26-L26)

### particula.aerosol_builder.AerosolBuilder.gas

*attribute* · *instance attribute*

```python
gas: GasData | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_builder.py#L25-L25)

### particula.aerosol_builder.AerosolBuilder.particles

*attribute* · *instance attribute*

```python
particles: ParticleData | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_builder.py#L24-L24)

### particula.aerosol_builder.AerosolBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.aerosol_builder.AerosolBuilder.__init__

*method*

```python
def __init__() -> None
```

Require particles, gas, and environment for every build.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_builder.py#L21-L26)

### particula.aerosol_builder.AerosolBuilder.build

*method*

```python
def build() -> Aerosol
```

Construct a fresh aggregate holding the exact supplied containers.

**Returns**

- (`Aerosol`) — Native Aerosol after structural and representation admission.

**Raises**

- `ValueError` — If any component is missing or invalid.
- `TypeError` — If a supplied component has the wrong container type.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_builder.py#L57-L72)

### particula.aerosol_builder.AerosolBuilder.set_environment

*method*

```python
def set_environment(environment: EnvironmentData, environment_units: str | None = None) -> AerosolBuilder
```

Retain environment; optional builder-protocol units are ignored.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_builder.py#L46-L55)

### particula.aerosol_builder.AerosolBuilder.set_gas

*method*

```python
def set_gas(gas: GasData, gas_units: str | None = None) -> AerosolBuilder
```

Retain full gas data; optional builder-protocol units are ignored.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_builder.py#L37-L44)

### particula.aerosol_builder.AerosolBuilder.set_particles

*method*

```python
def set_particles(particles: ParticleData, particles_units: str | None = None) -> AerosolBuilder
```

Retain particles; optional builder-protocol units are ignored.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_builder.py#L28-L35)

### particula.aerosol_builder.AerosolBuilder.check_keys

*method*

```python
def check_keys(parameters: dict[str, Any]) -> None
```

Inherited from: `particula.abc_builder.BuilderABC`

Check if the keys are present and valid.

**Parameters**

- `parameters` (`dict[str, Any]`) — The parameters dictionary to check.

**Raises**

- `ValueError` — If any required key is missing or if trying to set
an invalid parameter.

**Examples**

```py
builder = Builder()
builder.check_keys({
    "parameter1": 1,
    "parameter2": 2,
})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L53-L94)

### particula.aerosol_builder.AerosolBuilder.pre_build_check

*method*

```python
def pre_build_check() -> None
```

Inherited from: `particula.abc_builder.BuilderABC`

Check if all required attribute parameters are set before building.

**Raises**

- `ValueError` — If any required parameter is missing.

**Examples**

```py
builder = Builder()
builder.pre_build_check()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L132-L151)

### particula.aerosol_builder.AerosolBuilder.set_parameters

*method*

```python
def set_parameters(parameters: dict[str, Any]) -> BuilderABC
```

Inherited from: `particula.abc_builder.BuilderABC`

Set parameters from a dictionary, handling any '_units' suffix.

**Parameters**

- `parameters` (`dict[str, Any]`) — The parameters dictionary to set.

**Returns**

- `BuilderABC` (`BuilderABC`) — This builder object with the set parameters.

**Raises**

- `ValueError` — If any required key is missing.
- `Warning` — If using default units for any parameter.

**Examples**

```py
builder = Builder().set_parameters({
    "parameter1": 1,
    "parameter2": 2,
    "parameter2_units": "K",
})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L96-L130)
