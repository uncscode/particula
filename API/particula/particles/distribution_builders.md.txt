# particula.particles.distribution_builders

*module*

Builds distributions strategies based on the specified representation.

Currently, there are no parameters to set, but this is used for consistency
with other builder patterns in the codebase.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L1-L1)

## particula.particles.distribution_builders.MassBasedMovingBinBuilder

*class*

```python
class MassBasedMovingBinBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Builds and configures a MassBasedMovingBin instance for mass-based
distributions.

This builder requires no parameters, but is kept for consistency with
other builder patterns. Ensures a uniform interface for creating
MassBasedMovingBin objects.

Methods:
- build : Return a MassBasedMovingBin instance.

**Examples**

```py title="Example"
import particula as par

builder = par.particles.MassBasedMovingBinBuilder()
strategy = builder.build()
# strategy -> MassBasedMovingBin()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L16-L60)

### particula.particles.distribution_builders.MassBasedMovingBinBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.particles.distribution_builders.MassBasedMovingBinBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the MassBasedMovingBinBuilder.

Sets up the builder with no required parameters for creating a
MassBasedMovingBin strategy instance.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L37-L44)

### particula.particles.distribution_builders.MassBasedMovingBinBuilder.build

*method*

```python
def build() -> MassBasedMovingBin
```

Build and return a MassBasedMovingBin instance.

**Returns**

- `MassBasedMovingBin` (`MassBasedMovingBin`) — A strategy for mass-based particle distributions.

**Examples**

```py title="Build Example"
import particula as par
builder = par.particles.MassBasedMovingBinBuilder()
strategy = builder.build()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L46-L60)

### particula.particles.distribution_builders.MassBasedMovingBinBuilder.check_keys

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

### particula.particles.distribution_builders.MassBasedMovingBinBuilder.pre_build_check

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

### particula.particles.distribution_builders.MassBasedMovingBinBuilder.set_parameters

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

## particula.particles.distribution_builders.ParticleResolvedSpeciatedMassBuilder

*class*

```python
class ParticleResolvedSpeciatedMassBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Builds and configures a ParticleResolvedSpeciatedMass instance.

This builder requires no parameters, but follows the same pattern
to ensure uniform usage. ParticleResolvedSpeciatedMass is useful for
specific calculations when each particle's species composition must
be resolved individually.

Methods:
- build : Return a ParticleResolvedSpeciatedMass instance.

**Examples**

```py title="Example"
import particula as par
builder = par.particles.ParticleResolvedSpeciatedMassBuilder()
strategy = builder.build()
# strategy -> ParticleResolvedSpeciatedMass()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L154-L197)

### particula.particles.distribution_builders.ParticleResolvedSpeciatedMassBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.particles.distribution_builders.ParticleResolvedSpeciatedMassBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the ParticleResolvedSpeciatedMassBuilder.

Sets up the builder with no required parameters for creating a
ParticleResolvedSpeciatedMass strategy instance.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L174-L181)

### particula.particles.distribution_builders.ParticleResolvedSpeciatedMassBuilder.build

*method*

```python
def build() -> ParticleResolvedSpeciatedMass
```

Build and return a ParticleResolvedSpeciatedMass instance.

**Returns**

- `ParticleResolvedSpeciatedMass` (`ParticleResolvedSpeciatedMass`) — A strategy that resolves each particle's species composition independently.

**Examples**

```py title="Build Example"
import particula as par
builder = par.particles.ParticleResolvedSpeciatedMassBuilder()
strategy = builder.build()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L183-L197)

### particula.particles.distribution_builders.ParticleResolvedSpeciatedMassBuilder.check_keys

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

### particula.particles.distribution_builders.ParticleResolvedSpeciatedMassBuilder.pre_build_check

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

### particula.particles.distribution_builders.ParticleResolvedSpeciatedMassBuilder.set_parameters

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

## particula.particles.distribution_builders.RadiiBasedMovingBinBuilder

*class*

```python
class RadiiBasedMovingBinBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Builds and configures a RadiiBasedMovingBin instance for radius-based
distributions.

This builder requires no parameters, but is provided for consistency
with other builder patterns. Ensures a uniform interface for creating
RadiiBasedMovingBin objects.

Methods:
- build : Return a RadiiBasedMovingBin instance.

**Examples**

```py title="Example"
import particula as par
builder = par.particles.RadiiBasedMovingBinBuilder()
strategy = builder.build()
# strategy -> RadiiBasedMovingBin()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L63-L106)

### particula.particles.distribution_builders.RadiiBasedMovingBinBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.particles.distribution_builders.RadiiBasedMovingBinBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the RadiiBasedMovingBinBuilder.

Sets up the builder with no required parameters for creating a
RadiiBasedMovingBin strategy instance.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L83-L90)

### particula.particles.distribution_builders.RadiiBasedMovingBinBuilder.build

*method*

```python
def build() -> RadiiBasedMovingBin
```

Build and return a RadiiBasedMovingBin instance.

**Returns**

- `RadiiBasedMovingBin` (`RadiiBasedMovingBin`) — A strategy for radius-based particle distributions.

**Examples**

```py title="Build Example"
import particula as par
builder = par.particles.RadiiBasedMovingBinBuilder()
strategy = builder.build()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L92-L106)

### particula.particles.distribution_builders.RadiiBasedMovingBinBuilder.check_keys

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

### particula.particles.distribution_builders.RadiiBasedMovingBinBuilder.pre_build_check

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

### particula.particles.distribution_builders.RadiiBasedMovingBinBuilder.set_parameters

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

## particula.particles.distribution_builders.SpeciatedMassMovingBinBuilder

*class*

```python
class SpeciatedMassMovingBinBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Builds and configures a SpeciatedMassMovingBin instance for speciated
mass distributions.

This builder requires no parameters, but provides consistency with
other builder patterns and ensures a uniform interface for creating
SpeciatedMassMovingBin objects.

Methods:
- build : Return a SpeciatedMassMovingBin instance.

**Examples**

```py title="Example"
import particula as par
builder = par.particles.SpeciatedMassMovingBinBuilder()
strategy = builder.build()
# strategy -> SpeciatedMassMovingBin()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L109-L151)

### particula.particles.distribution_builders.SpeciatedMassMovingBinBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.particles.distribution_builders.SpeciatedMassMovingBinBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the SpeciatedMassMovingBinBuilder.

Sets up the builder with no required parameters for creating a
SpeciatedMassMovingBin strategy instance.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L129-L136)

### particula.particles.distribution_builders.SpeciatedMassMovingBinBuilder.build

*method*

```python
def build() -> SpeciatedMassMovingBin
```

Build and return a SpeciatedMassMovingBin instance.

**Returns**

- `SpeciatedMassMovingBin` (`SpeciatedMassMovingBin`) — A strategy for speciated mass distributions.

**Examples**

```py title="Build Example"
builder = SpeciatedMassMovingBinBuilder()
strategy = builder.build()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_builders.py#L138-L151)

### particula.particles.distribution_builders.SpeciatedMassMovingBinBuilder.check_keys

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

### particula.particles.distribution_builders.SpeciatedMassMovingBinBuilder.pre_build_check

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

### particula.particles.distribution_builders.SpeciatedMassMovingBinBuilder.set_parameters

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
