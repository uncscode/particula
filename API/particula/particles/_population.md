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

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/_population.py#L40-L45)

## particula.particles._population.radius_node_widths

*function*

```python
def radius_node_widths(radius: NDArray[np.float64]) -> NDArray[np.float64]
```

Trapezoidal integration widths (m) for an admitted 1-D radius grid.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/_population.py#L30-L37)

## particula.particles._population.slot_density

*function*

```python
def slot_density(particles: ParticleData) -> NDArray[np.float64]
```

Convert admitted extensive bins, slots or radius ordinates to density.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/_population.py#L12-L14)

## particula.particles._population.write_binned_density

*function*

```python
def write_binned_density(particles: ParticleData, density: NDArray[np.float64]) -> None
```

Commit admitted bin densities or PDF ordinates to existing storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/_population.py#L17-L27)
