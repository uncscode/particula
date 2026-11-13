# particula.dynamics.condensation.condensation_builder.native_configuration

*module*

Process-owned native configuration shared by condensation builders.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L1-L1)

## particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin

*class*

```python
class BuilderNativeCondensationMixin
```

Carry native strategy configuration through fluent and factory setup.

Legacy construction may omit every field. Explicit native calls validate
required configuration against current data at execution, including names
and lane bounds. Builders retain process strategy identities, never data.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L13-L127)

### particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin._NATIVE_KEYS

*attribute* · *class attribute* · *instance attribute*

```python
_NATIVE_KEYS = ('species_map', 'activity_strategy', 'surface_strategy', 'vapor_pressure_strategy', 'skip_partitioning_indices')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L21-L27)

### particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin.activity_strategy

*attribute* · *instance attribute*

```python
activity_strategy: ActivityStrategy | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L32-L32)

### particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin.skip_partitioning_indices

*attribute* · *instance attribute*

```python
skip_partitioning_indices: tuple[int, ...] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L37-L37)

### particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin.species_map

*attribute* · *instance attribute*

```python
species_map: CondensationSpeciesMap | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L31-L31)

### particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin.surface_strategy

*attribute* · *instance attribute*

```python
surface_strategy: SurfaceStrategy | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L33-L33)

### particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin.vapor_pressure_strategy

*attribute* · *instance attribute*

```python
vapor_pressure_strategy: VaporPressureStrategy | tuple[VaporPressureStrategy, ...] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L34-L36)

### particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize optional native configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L29-L37)

### particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin._load_native_parameters

*method*

```python
def _load_native_parameters(parameters: dict[str, Any]) -> dict[str, Any]
```

Load optional native keys and return remaining builder parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L107-L123)

### particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin._native_strategy_kwargs

*method*

```python
def _native_strategy_kwargs() -> dict[str, Any]
```

Return process configuration without constructing data or facades.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L125-L127)

### particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin.set_activity_strategy

*method*

```python
def set_activity_strategy(activity_strategy: ActivityStrategy | None) -> Self
```

Set the activity model that reads the full particle composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L52-L63)

### particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin.set_skip_partitioning_indices

*method*

```python
def set_skip_partitioning_indices(indices: Sequence[int] | None) -> Self
```

Detach full-gas skip indices; execution checks lane bounds.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L98-L105)

### particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin.set_species_map

*method*

```python
def set_species_map(species_map: CondensationSpeciesMap | None) -> Self
```

Set immutable full-gas provenance and explicit transfer pairs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L39-L50)

### particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin.set_surface_strategy

*method*

```python
def set_surface_strategy(surface_strategy: SurfaceStrategy | None) -> Self
```

Set the surface model that reads the full particle composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L65-L74)

### particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin.set_vapor_pressure_strategy

*method*

```python
def set_vapor_pressure_strategy(vapor_pressure_strategy: VaporPressureStrategy | Sequence[VaporPressureStrategy] | None) -> Self
```

Set one vector model or a detached ordered model sequence.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L76-L96)
