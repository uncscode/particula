# particula.execution.inverse_condensation

*module*

Differentiable fixed-step growth in closed single-material boxes.

This concrete research path assumes zero equilibrium vapor, unit accommodation,
constant diffusivity and positive particle masses. Each occupied slot represents
a monodisperse component. Independent boxes may share initial guesses.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_condensation.py#L1-L1)

## particula.execution.inverse_condensation._advance

*function*

```python
def _advance(current: FloatMatrix, gas: FloatVector, factor: wp.float64, mean_path: wp.float64, diffusion: wp.float64, dt: wp.float64, result: FloatMatrix)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_condensation.py#L51-L69)

## particula.execution.inverse_condensation._growth_inputs

*function*

```python
def _growth_inputs(initial_um, number_density, vapor, density, mean_path, diffusivity, duration, steps, segment_steps, adjoint)
```

Validate the interior growth domain before allocating device state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_condensation.py#L84-L151)

## particula.execution.inverse_condensation._mass_from_radius

*function*

```python
def _mass_from_radius(radius: FloatMatrix, factor: wp.float64, mass: FloatMatrix)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_condensation.py#L25-L33)

## particula.execution.inverse_condensation._observe

*function*

```python
def _observe(mass: FloatMatrix, factor: wp.float64, result: FloatMatrix)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_condensation.py#L72-L81)

## particula.execution.inverse_condensation._reservoir

*function*

```python
def _reservoir(initial: FloatMatrix, current: FloatMatrix, counts: FloatMatrix, initial_gas: FloatVector, gas: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_condensation.py#L36-L48)

## particula.execution.inverse_condensation.terminal_radius

*function*

```python
def terminal_radius(initial_um, number_density, vapor, *, density, mean_path, diffusivity, duration, steps, device, adjoint=None, segment_steps=None)
```

Evaluate wet radii (µm) and optional vector-Jacobian product.

initial_um and number_density (m⁻³) are positive box-by-component arrays.
vapor (kg/m³) has one entry per box. Density is kg/m³, molecular mean
free path is m, diffusivity is m²/s and duration is seconds. An optional
adjoint gives one terminal observation weight per component.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_condensation.py#L154-L257)
