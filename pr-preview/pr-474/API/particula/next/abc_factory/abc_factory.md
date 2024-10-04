# Abc Factory

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Next](./index.md#next) / Abc Factory

> Auto-generated documentation for [particula.next.abc_factory](https://github.com/uncscode/particula/blob/main/particula/next/abc_factory.py) module.

#### Attributes

- `BuilderT` - Define a generic type variable for the strategy type, to get good type hints: TypeVar('BuilderT')


## StrategyFactory

[Show source in abc_factory.py:18](https://github.com/uncscode/particula/blob/main/particula/next/abc_factory.py#L18)

Abstract base class for strategy factories.

#### Signature

```python
class StrategyFactory(ABC, Generic[BuilderT, StrategyT]): ...
```

#### See also

- [BuilderT](#buildert)
- [StrategyT](#strategyt)

### StrategyFactory().get_builders

[Show source in abc_factory.py:23](https://github.com/uncscode/particula/blob/main/particula/next/abc_factory.py#L23)

Returns the mapping of strategy types to builder instances.

#### Signature

```python
@abstractmethod
def get_builders(self) -> Dict[str, BuilderT]: ...
```

#### See also

- [BuilderT](#buildert)

### StrategyFactory().get_strategy

[Show source in abc_factory.py:29](https://github.com/uncscode/particula/blob/main/particula/next/abc_factory.py#L29)

Generic factory method to create strategies.

#### Signature

```python
def get_strategy(
    self, strategy_type: str, parameters: Optional[Dict[str, Any]] = None
) -> StrategyT: ...
```

#### See also

- [StrategyT](#strategyt)
