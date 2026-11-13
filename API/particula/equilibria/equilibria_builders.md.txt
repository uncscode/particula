# particula.equilibria.equilibria_builders

*module*

Builder implementations for equilibria strategies.

Provides fluent builders to configure equilibria strategies with
validation and deterministic parameter handling.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_builders.py#L1-L1)

## particula.equilibria.equilibria_builders.LiquidVaporPartitioningBuilder

*class*

```python
class LiquidVaporPartitioningBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Builder for :class:`LiquidVaporPartitioningStrategy`.

Supports fluent configuration with validation for water activity and
deterministic parameter handling for factory usage.

**Examples**

```pycon
>>> builder = LiquidVaporPartitioningBuilder()
>>> strategy = builder.set_water_activity(0.8).build()
>>> strategy.water_activity
0.8
```

```pycon
>>> builder = LiquidVaporPartitioningBuilder()
>>> strategy = builder.set_parameters({"water_activity": 0.65}).build()
>>> strategy.water_activity
0.65
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_builders.py#L17-L100)

### particula.equilibria.equilibria_builders.LiquidVaporPartitioningBuilder._water_activity

*attribute* · *instance attribute*

```python
_water_activity: float | None = 0.5
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_builders.py#L38-L38)

### particula.equilibria.equilibria_builders.LiquidVaporPartitioningBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.equilibria.equilibria_builders.LiquidVaporPartitioningBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize builder with default water activity.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_builders.py#L35-L38)

### particula.equilibria.equilibria_builders.LiquidVaporPartitioningBuilder.build

*method*

```python
def build() -> LiquidVaporPartitioningStrategy
```

Build the configured :class:`LiquidVaporPartitioningStrategy`.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_builders.py#L93-L100)

### particula.equilibria.equilibria_builders.LiquidVaporPartitioningBuilder.pre_build_check

*method*

```python
def pre_build_check() -> None
```

Ensure builder is ready to build a strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_builders.py#L87-L91)

### particula.equilibria.equilibria_builders.LiquidVaporPartitioningBuilder.set_parameters

*method*

```python
def set_parameters(parameters: dict[str, Any]) -> Self
```

Set parameters from a dictionary with strict key validation.

Only ``{"water_activity"}`` is accepted. Unknown keys, including
unit-suffixed variants, raise ``ValueError`` to keep behavior
deterministic for factories.

**Parameters**

- `parameters` (`dict[str, Any]`) — Parameter dictionary.

**Returns**

- (`Self`) — Self for method chaining.

**Raises**

- `ValueError` — If an unknown key is provided or value is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_builders.py#L58-L85)

### particula.equilibria.equilibria_builders.LiquidVaporPartitioningBuilder.set_water_activity

*method*

```python
def set_water_activity(value: float) -> Self
```

Set the target water activity.

**Parameters**

- `value` (`float`) — Water activity in range [0, 1].

**Returns**

- (`Self`) — Self for method chaining.

**Raises**

- `ValueError` — If ``value`` is outside the inclusive [0, 1] range.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/equilibria_builders.py#L40-L56)

### particula.equilibria.equilibria_builders.LiquidVaporPartitioningBuilder.check_keys

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
