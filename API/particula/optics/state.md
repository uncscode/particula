# particula.optics.state

*module*

Bulk and individual optics from native particle state.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L1-L1)

## particula.optics.state.OpticalConfig

*class* · *dataclass*

```python
class OpticalConfig
```

Prescribed optical inputs for homogeneous spheres.

Wavelengths are ordered vacuum wavelengths [m]. Supply an absolute
particle index (scalar or per wavelength), or a species index table
(wavelength, material) in native particle-material order. Medium index
is real, positive and constant across the spectrum.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L50-L142)

### particula.optics.state.OpticalConfig.medium_index

*attribute* · *instance attribute*

```python
medium_index: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L61-L61)

### particula.optics.state.OpticalConfig.particle_index

*attribute* · *instance attribute*

```python
particle_index: tuple[complex, ...] | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L62-L62)

### particula.optics.state.OpticalConfig.species_index

*attribute* · *instance attribute*

```python
species_index: tuple[tuple[complex, ...], ...] | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L63-L63)

### particula.optics.state.OpticalConfig.wavelength

*attribute* · *instance attribute*

```python
wavelength: tuple[float, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L60-L60)

### particula.optics.state.OpticalConfig.__init__

*method*

```python
def __init__(wavelength, *, particle_index=None, species_index=None, medium_index=1.0)
```

Validate and freeze wavelengths and the chosen index model.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L65-L116)

### particula.optics.state.OpticalConfig.metadata

*method*

```python
def metadata()
```

Return JSON-compatible optical inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L126-L142)

### particula.optics.state.OpticalConfig.validate_materials

*method*

```python
def validate_materials(materials)
```

Check the species table against the native material axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L118-L124)

## particula.optics.state._index

*function*

```python
def _index(values)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L16-L24)

## particula.optics.state.lorentz_lorenz

*function*

```python
def lorentz_lorenz(index)
```

Convert absolute refractive index to Lorentz–Lorenz polarizability.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L27-L30)

## particula.optics.state.mix_refractive_index

*function*

```python
def mix_refractive_index(index, volume_fraction)
```

Mix absolute indices along the last axis on a volume basis.

The fractions are dimensionless, nonnegative and sum to one.
Both inputs broadcast with their common final material axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L33-L47)

## particula.optics.state.particle_optics

*function*

```python
def particle_optics(particles, config)
```

Return per-particle Mie efficiencies and cross sections.

Arrays have axes (box, wavelength, slot). Empty slots have zero radius
for the optical calculation. Discrete and resolved sizes come from
material mass and density. Continuous PDFs use their explicit nodes.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L145-L185)

## particula.optics.state.population_optics

*function*

```python
def population_optics(particles, config) -> PopulationOptics
```

Evaluate live CPU particle state into (box, wavelength) results.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L188-L217)
