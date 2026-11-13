# particula.dynamics.condensation.condensation_builder.condensation_builder_mixin

*module*

Mixin classes for Condensation strategy builders.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L1-L1)

## particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L14-L14)

## particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin

*class*

```python
class BuilderAccommodationCoefficientMixin
```

Mixin to set the mass accommodation coefficient.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L49-L73)

### particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin.accommodation_coefficient

*attribute* · *instance attribute*

```python
accommodation_coefficient: Optional[Union[float, NDArray[np.float64]]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L57-L59)

### particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize the accommodation coefficient mixin.

Sets the accommodation_coefficient attribute to None.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L52-L59)

### particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin.set_accommodation_coefficient

*method*

```python
def set_accommodation_coefficient(accommodation_coefficient: Union[float, NDArray[np.float64]], accommodation_coefficient_units: Optional[str] = None)
```

Set the dimensionless mass accommodation coefficient.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L61-L73)

## particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin

*class*

```python
class BuilderDiffusionCoefficientMixin
```

Mixin to set a diffusion coefficient in m^2/s.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L17-L46)

### particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin.diffusion_coefficient

*attribute* · *instance attribute*

```python
diffusion_coefficient: Optional[Union[float, NDArray[np.float64]]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L25-L27)

### particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize the diffusion coefficient mixin.

Sets the diffusion_coefficient attribute to None.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L20-L27)

### particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin.set_diffusion_coefficient

*method*

```python
def set_diffusion_coefficient(diffusion_coefficient: Union[float, NDArray[np.float64]], diffusion_coefficient_units: str)
```

Set the diffusion coefficient for the condensing species.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L29-L46)

## particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin

*class*

```python
class BuilderUpdateGasesMixin
```

Mixin to specify whether the gas phase should be updated.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L76-L93)

### particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin.update_gases

*attribute* · *instance attribute*

```python
update_gases: bool = True
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L84-L84)

### particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize the update gases mixin.

Sets the update_gases flag to True by default.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L79-L84)

### particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin.set_update_gases

*method*

```python
def set_update_gases(update_gases: bool, update_gases_units: Optional[str] = None)
```

Set the flag controlling gas-phase updates.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L86-L93)
