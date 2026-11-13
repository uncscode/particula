# particula.dynamics.wall_loss.wall_loss_factories

*module*

Wall loss strategy factory.

Provides a factory for creating neutral or charged wall loss strategies by
name using registered builders.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_factories.py#L1-L1)

## particula.dynamics.wall_loss.wall_loss_factories.BuilderType

*attribute* · *module attribute*

```python
BuilderType = Union[SphericalWallLossBuilder, RectangularWallLossBuilder, ChargedWallLossBuilder]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_factories.py#L16-L20)

## particula.dynamics.wall_loss.wall_loss_factories.WallLossFactory

*class*

```python
class WallLossFactory(StrategyFactoryABC)
```

Bases: `particula.abc_factory.StrategyFactoryABC`

Factory for wall loss strategies.

Builds wall loss strategies using registered builders and the
:class:`StrategyFactoryABC` workflow.

> **Supported strategy types**
>
> - "spherical": ``SphericalWallLossStrategy`` for spherical chambers.
> - "rectangular": ``RectangularWallLossStrategy`` for box chambers.
> - "charged": ``ChargedWallLossStrategy`` with image-charge
>   enhancement and optional electric-field drift; requires geometry
>   and matching size parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_factories.py#L23-L82)

### particula.dynamics.wall_loss.wall_loss_factories.WallLossFactory.get_builders

*method*

```python
def get_builders() -> Dict[str, BuilderType]
```

Return available wall loss builders keyed by strategy name.

**Returns**

- (`Dict[str, BuilderType]`) — Dict[str, BuilderType]: Mapping of strategy type to builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_factories.py#L37-L47)

### particula.dynamics.wall_loss.wall_loss_factories.WallLossFactory.get_strategy

*method*

```python
def get_strategy(strategy_type: str, parameters: Optional[Dict[str, Any]] = None)
```

Create a wall loss strategy using its corresponding builder.

**Parameters**

- `strategy_type` (`str`) — Strategy name ("spherical", "rectangular", or "charged").
- `parameters` (`Optional[Dict[str, Any]]`) (default: `None`) — Optional parameter mapping for the builder. Include geometry keys (``chamber_radius`` or ``chamber_dimensions``) plus optional ``wall_potential`` and ``wall_electric_field`` entries, and ``distribution_type`` when needed.

**Returns**

- `result` (`WallLossStrategy`) — Built wall loss strategy instance.

**Raises**

- `ValueError` — If ``strategy_type`` is unknown or parameters are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_factories.py#L49-L82)
