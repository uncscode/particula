# Abc Builder

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Next](./index.md#next) / Abc Builder

> Auto-generated documentation for [particula.next.abc_builder](https://github.com/uncscode/particula/blob/main/particula/next/abc_builder.py) module.

## BuilderABC

[Show source in abc_builder.py:23](https://github.com/uncscode/particula/blob/main/particula/next/abc_builder.py#L23)

Abstract base class for builders with common methods to check keys and
set parameters from a dictionary.

#### Attributes

- `required_parameters` - List of required parameters for the builder.

#### Methods

- `check_keys` *parameters* - Check if the keys you want to set are
present in the parameters dictionary.
- `set_parameters` *parameters* - Set parameters from a dictionary including
optional suffix for units as '_units'.
- `pre_build_check()` - Check if all required attribute parameters are set
before building.
- `build` *abstract* - Build and return the strategy object.

#### Raises

- `ValueError` - If any required key is missing during check_keys or
pre_build_check, or if trying to set an invalid parameter.
- `Warning` - If using default units for any parameter.

#### References

This module also defines mixin classes for the Builder classes to set
some optional method to be used in the Builder classes.
[Mixin Wikipedia](https://en.wikipedia.org/wiki/Mixin)

#### Signature

```python
class BuilderABC(ABC):
    def __init__(self, required_parameters: Optional[list[str]] = None): ...
```

### BuilderABC().build

[Show source in abc_builder.py:133](https://github.com/uncscode/particula/blob/main/particula/next/abc_builder.py#L133)

Build and return the strategy object with the set parameters.

#### Returns

- `strategy` - The built strategy object.

#### Signature

```python
@abstractmethod
def build(self) -> Any: ...
```

### BuilderABC().check_keys

[Show source in abc_builder.py:53](https://github.com/uncscode/particula/blob/main/particula/next/abc_builder.py#L53)

Check if the keys are present and valid.

#### Arguments

- `parameters` - The parameters dictionary to check.

#### Raises

- `ValueError` - If any required key is missing or if trying to set an
invalid parameter.

#### Signature

```python
def check_keys(self, parameters: dict[str, Any]): ...
```

### BuilderABC().pre_build_check

[Show source in abc_builder.py:118](https://github.com/uncscode/particula/blob/main/particula/next/abc_builder.py#L118)

Check if all required attribute parameters are set before building.

#### Raises

- `ValueError` - If any required parameter is missing.

#### Signature

```python
def pre_build_check(self): ...
```

### BuilderABC().set_parameters

[Show source in abc_builder.py:90](https://github.com/uncscode/particula/blob/main/particula/next/abc_builder.py#L90)

Set parameters from a dictionary including optional suffix for
units as '_units'.

#### Arguments

- `parameters` - The parameters dictionary to set.

#### Returns

- `self` - The builder object with the set parameters.

#### Raises

- `ValueError` - If any required key is missing.
- `Warning` - If using default units for any parameter.

#### Signature

```python
def set_parameters(self, parameters: dict[str, Any]): ...
```
