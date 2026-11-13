# particula.dynamics.coagulation.coagulation_strategy._native_validation

*module*

Read-only admission for explicitly tagged native coagulation state.

Public kernels require explicit population metadata. Kernel evaluation
uses a private material-only view for positive-mass lanes, then restores the
full slot matrix with zero rows/columns for zero-mass lanes. This does not
select collisions, mutate a population, or assign it a different representation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_validation.py#L1-L1)

## particula.dynamics.coagulation.coagulation_strategy._native_validation._KernelMethod

*attribute* · *module attribute*

```python
_KernelMethod = TypeVar('_KernelMethod', bound=Callable[..., Any])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_validation.py#L18-L18)

## particula.dynamics.coagulation.coagulation_strategy._native_validation._evaluate_material_kernel

*function*

```python
def _evaluate_material_kernel(strategy: Any, material: ParticleData, temperature: float, pressure: float) -> Any
```

Evaluate an admitted combination's child without public re-admission.

Decorated built-ins expose their material law privately. Custom kernels
retain their existing direct material-input contract. This helper is used
only inside a combined kernel after the outer population/control checks.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_validation.py#L195-L208)

## particula.dynamics.coagulation.coagulation_strategy._native_validation.native_kernel

*function*

```python
def native_kernel(method: _KernelMethod) -> _KernelMethod
```

Admit native state and retain the full kernel slot layout.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_validation.py#L118-L163)

## particula.dynamics.coagulation.coagulation_strategy._native_validation.positive_scalar

*function*

```python
def positive_scalar(value: Any, name: str, *, zero_allowed: bool = False) -> None
```

Require a finite physical scalar, excluding boolean/array coercions.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_validation.py#L21-L32)

## particula.dynamics.coagulation.coagulation_strategy._native_validation.rate_radius

*function*

```python
def rate_radius(particle: ParticleData) -> NDArray[np.float64]
```

Use the explicit PDF coordinate, otherwise the physical particle size.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_validation.py#L186-L192)

## particula.dynamics.coagulation.coagulation_strategy._native_validation.validate_controls

*function*

```python
def validate_controls(strategy: Any, kind: str, visiting: set[int] | None = None) -> None
```

Revalidate mutable mechanism controls and composed distribution modes.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_validation.py#L68-L115)

## particula.dynamics.coagulation.coagulation_strategy._native_validation.validate_particle

*function*

```python
def validate_particle(particle: ParticleData, distribution_type: str) -> None
```

Check current representation, layout, physics and strategy agreement.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_validation.py#L35-L65)

## particula.dynamics.coagulation.coagulation_strategy._native_validation.validate_rate_kernel

*function*

```python
def validate_rate_kernel(particle: ParticleData, kernel: Any) -> NDArray[np.float64]
```

Require one finite nonnegative pair value per original slot pair.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_validation.py#L166-L183)
