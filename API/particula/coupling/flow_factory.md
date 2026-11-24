# particula.coupling.flow_factory

*module*

Particula builders for spatial flow and recorded forcing providers.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L1-L1)

## particula.coupling.flow_factory.DuneFlowBuilder

*class*

```python
class DuneFlowBuilder(SoundingFlowBuilder)
```

Bases: `particula.coupling.flow_factory.SoundingFlowBuilder`

Build the Gaussian dune with a vertical sounding.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L102-L142)

### particula.coupling.flow_factory.DuneFlowBuilder.height

*attribute* · *instance attribute*

```python
height = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L109-L109)

### particula.coupling.flow_factory.DuneFlowBuilder.lid

*attribute* · *instance attribute*

```python
lid = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L109-L109)

### particula.coupling.flow_factory.DuneFlowBuilder.width

*attribute* · *instance attribute*

```python
width = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L109-L109)

### particula.coupling.flow_factory.DuneFlowBuilder.sounding

*attribute* · *instance attribute*

```python
sounding = None
```

Inherited from: `particula.coupling.flow_factory.SoundingFlowBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L66-L66)

### particula.coupling.flow_factory.DuneFlowBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.coupling.flow_factory.DuneFlowBuilder.__init__

*method*

```python
def __init__()
```

Declare the parameters required by this provider.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L105-L109)

### particula.coupling.flow_factory.DuneFlowBuilder.build

*method*

```python
def build()
```

Construct terrain-following forcing.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L133-L142)

### particula.coupling.flow_factory.DuneFlowBuilder.set_height

*method*

```python
def set_height(value, units='m')
```

Set crest elevation.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L118-L121)

### particula.coupling.flow_factory.DuneFlowBuilder.set_lid

*method*

```python
def set_lid(value, units='m')
```

Set lid height above datum.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L128-L131)

### particula.coupling.flow_factory.DuneFlowBuilder.set_sounding

*method*

```python
def set_sounding(value)
```

Use either a steady profile or timestamped profiles.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L111-L116)

### particula.coupling.flow_factory.DuneFlowBuilder.set_width

*method*

```python
def set_width(value, units='m')
```

Set the Gaussian standard deviation.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L123-L126)

### particula.coupling.flow_factory.DuneFlowBuilder.check_keys

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

### particula.coupling.flow_factory.DuneFlowBuilder.pre_build_check

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

### particula.coupling.flow_factory.DuneFlowBuilder.set_parameters

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

## particula.coupling.flow_factory.FlowFactory

*class*

```python
class FlowFactory(StrategyFactoryABC)
```

Bases: `particula.abc_factory.StrategyFactoryABC`

Construct an independent builder for each supported flow provider.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L173-L185)

### particula.coupling.flow_factory.FlowFactory.get_builders

*method*

```python
def get_builders()
```

Map flow names to fresh builders.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L176-L185)

### particula.coupling.flow_factory.FlowFactory.get_strategy

*method*

```python
def get_strategy(strategy_type: str, parameters: Optional[Dict[str, Any]] = None) -> StrategyT
```

Inherited from: `particula.abc_factory.StrategyFactoryABC`

Create a strategy instance using its corresponding builder.

**Parameters**

- `strategy_type` (`str`) — Name of the strategy to build.
- `parameters` (`Dict[str, Any]`) (default: `None`) — Dictionary of parameters to configure the chosen builder.

**Returns**

- `StrategyT` (`StrategyT`) — The built strategy object corresponding to the specified type.

**Raises**

- `ValueError` — If the `strategy_type` is unknown, or if any required
parameter is invalid/missing for the chosen builder.

**Examples**

```py title="Strategy Creation Example"
my_factory = SomeStrategyFactory()
my_strategy = my_factory.get_strategy(
    "desired_strategy", {"param_x": 42}
)
# my_strategy is now an instance configured with param_x=42
```

> **References**
>
> - "Factory Method Pattern,"
> [Wikipedia](https://en.wikipedia.org/wiki/Factory_method_pattern)

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_factory.py#L69-L110)

## particula.coupling.flow_factory.GridReplayBuilder

*class*

```python
class GridReplayBuilder(TrajectoryReplayBuilder)
```

Bases: `particula.coupling.flow_factory.TrajectoryReplayBuilder`

Load accepted grid forcing from its numerical archive.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L164-L170)

### particula.coupling.flow_factory.GridReplayBuilder.path

*attribute* · *instance attribute*

```python
path = None
```

Inherited from: `particula.coupling.flow_factory.TrajectoryReplayBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L151-L151)

### particula.coupling.flow_factory.GridReplayBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.coupling.flow_factory.GridReplayBuilder.build

*method*

```python
def build()
```

Validate and return exact gridded forcing.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L167-L170)

### particula.coupling.flow_factory.GridReplayBuilder.__init__

*method*

```python
def __init__()
```

Inherited from: `particula.coupling.flow_factory.TrajectoryReplayBuilder`

Declare the parameters required by this provider.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L148-L151)

### particula.coupling.flow_factory.GridReplayBuilder.set_path

*method*

```python
def set_path(value)
```

Inherited from: `particula.coupling.flow_factory.TrajectoryReplayBuilder`

Select the forcing archive.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L153-L156)

### particula.coupling.flow_factory.GridReplayBuilder.check_keys

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

### particula.coupling.flow_factory.GridReplayBuilder.pre_build_check

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

### particula.coupling.flow_factory.GridReplayBuilder.set_parameters

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

## particula.coupling.flow_factory.SoundingFlowBuilder

*class*

```python
class SoundingFlowBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Build from a normalized SI sounding object or CSV path.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L60-L78)

### particula.coupling.flow_factory.SoundingFlowBuilder.sounding

*attribute* · *instance attribute*

```python
sounding = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L66-L66)

### particula.coupling.flow_factory.SoundingFlowBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.coupling.flow_factory.SoundingFlowBuilder.__init__

*method*

```python
def __init__()
```

Declare the parameters required by this provider.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L63-L66)

### particula.coupling.flow_factory.SoundingFlowBuilder.build

*method*

```python
def build()
```

Construct horizontally uniform profile forcing.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L75-L78)

### particula.coupling.flow_factory.SoundingFlowBuilder.set_sounding

*method*

```python
def set_sounding(value)
```

Supply a sounding or normalized CSV.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L68-L73)

### particula.coupling.flow_factory.SoundingFlowBuilder.check_keys

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

### particula.coupling.flow_factory.SoundingFlowBuilder.pre_build_check

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

### particula.coupling.flow_factory.SoundingFlowBuilder.set_parameters

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

## particula.coupling.flow_factory.SoundingSequenceBuilder

*class*

```python
class SoundingSequenceBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Construct time-indexed profile forcing with explicit provenance.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L81-L99)

### particula.coupling.flow_factory.SoundingSequenceBuilder.sequence

*attribute* · *instance attribute*

```python
sequence = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L87-L87)

### particula.coupling.flow_factory.SoundingSequenceBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.coupling.flow_factory.SoundingSequenceBuilder.__init__

*method*

```python
def __init__()
```

Require the timestamped set of normalized SI profiles.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L84-L87)

### particula.coupling.flow_factory.SoundingSequenceBuilder.build

*method*

```python
def build()
```

Construct time-dependent spatial forcing.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L96-L99)

### particula.coupling.flow_factory.SoundingSequenceBuilder.set_sequence

*method*

```python
def set_sequence(value)
```

Bind a SoundingSequence.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L89-L94)

### particula.coupling.flow_factory.SoundingSequenceBuilder.check_keys

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

### particula.coupling.flow_factory.SoundingSequenceBuilder.pre_build_check

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

### particula.coupling.flow_factory.SoundingSequenceBuilder.set_parameters

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

## particula.coupling.flow_factory.TrajectoryReplayBuilder

*class*

```python
class TrajectoryReplayBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Build from a native parcel CSV and its sibling metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L145-L161)

### particula.coupling.flow_factory.TrajectoryReplayBuilder.path

*attribute* · *instance attribute*

```python
path = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L151-L151)

### particula.coupling.flow_factory.TrajectoryReplayBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.coupling.flow_factory.TrajectoryReplayBuilder.__init__

*method*

```python
def __init__()
```

Declare the parameters required by this provider.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L148-L151)

### particula.coupling.flow_factory.TrajectoryReplayBuilder.build

*method*

```python
def build()
```

Load a trajectory by its recorded identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L158-L161)

### particula.coupling.flow_factory.TrajectoryReplayBuilder.set_path

*method*

```python
def set_path(value)
```

Select the forcing archive.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L153-L156)

### particula.coupling.flow_factory.TrajectoryReplayBuilder.check_keys

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

### particula.coupling.flow_factory.TrajectoryReplayBuilder.pre_build_check

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

### particula.coupling.flow_factory.TrajectoryReplayBuilder.set_parameters

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

## particula.coupling.flow_factory.UniformFlowBuilder

*class*

```python
class UniformFlowBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Build a spatially uniform flow with conversion to SI.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L24-L57)

### particula.coupling.flow_factory.UniformFlowBuilder.pressure

*attribute* · *instance attribute*

```python
pressure = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L30-L30)

### particula.coupling.flow_factory.UniformFlowBuilder.temperature

*attribute* · *instance attribute*

```python
temperature = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L30-L30)

### particula.coupling.flow_factory.UniformFlowBuilder.vapor

*attribute* · *instance attribute*

```python
vapor = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L30-L30)

### particula.coupling.flow_factory.UniformFlowBuilder.velocity

*attribute* · *instance attribute*

```python
velocity = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L30-L30)

### particula.coupling.flow_factory.UniformFlowBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.coupling.flow_factory.UniformFlowBuilder.__init__

*method*

```python
def __init__()
```

Declare the parameters required by this provider.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L27-L30)

### particula.coupling.flow_factory.UniformFlowBuilder.build

*method*

```python
def build()
```

Construct validated constant fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L52-L57)

### particula.coupling.flow_factory.UniformFlowBuilder.set_pressure

*method*

```python
def set_pressure(value, units='Pa')
```

Set the total pressure.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L42-L45)

### particula.coupling.flow_factory.UniformFlowBuilder.set_temperature

*method*

```python
def set_temperature(value, units='K')
```

Set the air temperature.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L37-L40)

### particula.coupling.flow_factory.UniformFlowBuilder.set_vapor

*method*

```python
def set_vapor(value, units='kg/kg')
```

Set water mass per kg dry air.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L47-L50)

### particula.coupling.flow_factory.UniformFlowBuilder.set_velocity

*method*

```python
def set_velocity(value, units='m/s')
```

Set the east, north and vertical wind components.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L32-L35)

### particula.coupling.flow_factory.UniformFlowBuilder.check_keys

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

### particula.coupling.flow_factory.UniformFlowBuilder.pre_build_check

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

### particula.coupling.flow_factory.UniformFlowBuilder.set_parameters

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

## particula.coupling.flow_factory._convert

*function*

```python
def _convert(units, target, *, value)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_factory.py#L16-L21)
