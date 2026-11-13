# particula.abc_builder

*module*

Abstract Base Class for Builder classes.

> **References**
>
> - Builder Pattern : https://refactoring.guru/design-patterns/builder

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L1-L1)

## particula.abc_builder.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L12-L12)

## particula.abc_builder.BuilderABC

*class*

```python
class BuilderABC(ABC)
```

Bases: `ABC`

Abstract base class for builders to check keys and set parameters.

**Attributes**

- `required_parameters` — List of required parameters for the builder.

**Raises**

- `ValueError` — If any required key is missing during check_keys or
pre_build_check, or if trying to set an invalid parameter.
- `Warning` — If using default units for any parameter.

**Examples**

```py
class MyBuilder(BuilderABC):
    def set_parameter1(self, value, units=None):
        ...
    def set_parameter2(self, value, units=None):
        ...
    def build(self):
        return SomeStrategy()

strategy = (
    MyBuilder()
    .set_parameters1(10, 'm')
    .set_parameters2(20, 's')
    .build()
)
```

> **References**
>
> - "Builder Pattern,"
> [Refactoring Guru](https://refactoring.guru/design-patterns/builder)

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L15-L165)

### particula.abc_builder.BuilderABC.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.abc_builder.BuilderABC.__init__

*method*

```python
def __init__(required_parameters: Optional[list[str]] = None) -> None
```

Initialize builder with required parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L49-L51)

### particula.abc_builder.BuilderABC.build

*method* · *abstract*

```python
def build() -> Any
```

Build and return the strategy object with the set parameters.

**Returns**

- `Any` (`Any`) — The built strategy object.

**Examples**

```py
builder = Builder()
strategy = builder.build()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L153-L165)

### particula.abc_builder.BuilderABC.check_keys

*method*

```python
def check_keys(parameters: dict[str, Any]) -> None
```

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

### particula.abc_builder.BuilderABC.pre_build_check

*method*

```python
def pre_build_check() -> None
```

Check if all required attribute parameters are set before building.

**Raises**

- `ValueError` — If any required parameter is missing.

**Examples**

```py
builder = Builder()
builder.pre_build_check()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L132-L151)

### particula.abc_builder.BuilderABC.set_parameters

*method*

```python
def set_parameters(parameters: dict[str, Any]) -> BuilderABC
```

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
