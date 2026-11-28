# particula.particles._population

*module*

Internal conversions between raw storage and physical population measures.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/_population.py#L1-L1)

## particula.particles._population.number_density_weights

*function*

```python
def number_density_weights(particles: ParticleData) -> NDArray[np.float64]
```

Return detached (box, slot) contributions to number density in m⁻³.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/_population.py#L41-L46)

## particula.particles._population.radius_node_widths

*function*

```python
def radius_node_widths(radius: NDArray[np.float64]) -> NDArray[np.float64]
```

Trapezoidal integration widths (m) for an admitted 1-D radius grid.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/_population.py#L31-L38)

## particula.particles._population.slot_density

*function*

```python
def slot_density(particles: ParticleData) -> NDArray[np.float64]
```

Read admitted storage as m⁻³ (bins/slots) or m⁻⁴ (radius PDF).

Binned storage is already a density. Treat the result as read-only.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/_population.py#L12-L19)

## particula.particles._population.write_binned_density

*function*

```python
def write_binned_density(particles: ParticleData, density: NDArray[np.float64]) -> None
```

Commit admitted bin densities or PDF ordinates to existing storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/_population.py#L22-L28)
