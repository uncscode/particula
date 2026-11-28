# particula.particles.properties.number_spectrum

*module*

Number-density spectra on caller-supplied diameter edges.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/number_spectrum.py#L1-L1)

## particula.particles.properties.number_spectrum.NumberSpectrum

*class*

```python
class NumberSpectrum(NamedTuple)
```

Bases: `NamedTuple`

Integrated densities [m⁻³] on the last axis and two per-row tails.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/number_spectrum.py#L12-L17)

### particula.particles.properties.number_spectrum.NumberSpectrum.above_range

*attribute* · *instance attribute*

```python
above_range: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/number_spectrum.py#L17-L17)

### particula.particles.properties.number_spectrum.NumberSpectrum.below_range

*attribute* · *instance attribute*

```python
below_range: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/number_spectrum.py#L16-L16)

### particula.particles.properties.number_spectrum.NumberSpectrum.bin_number_density

*attribute* · *instance attribute*

```python
bin_number_density: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/number_spectrum.py#L15-L15)

## particula.particles.properties.number_spectrum._bin_values

*function*

```python
def _bin_values(values, edges)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/number_spectrum.py#L53-L59)

## particula.particles.properties.number_spectrum._edges

*function*

```python
def _edges(diameter_edges: ArrayLike) -> NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/number_spectrum.py#L29-L38)

## particula.particles.properties.number_spectrum._log_width

*function*

```python
def _log_width(low, high)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/number_spectrum.py#L41-L50)

## particula.particles.properties.number_spectrum._real

*function*

```python
def _real(values: ArrayLike) -> NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/number_spectrum.py#L20-L26)

## particula.particles.properties.number_spectrum.get_bin_number_density

*function*

```python
def get_bin_number_density(number_per_decade: ArrayLike, diameter_edges: ArrayLike) -> NDArray[np.float64]
```

Integrate dN/dlog10D [m⁻³ per decade] over each diameter interval.

The shared 1-D edges are in metres. Returns number density per bin [m⁻³]
with the bin axis last and all leading dimensions preserved.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/number_spectrum.py#L76-L87)

## particula.particles.properties.number_spectrum.get_binned_number_spectrum

*function*

```python
def get_binned_number_spectrum(diameter: ArrayLike, number_density: ArrayLike, diameter_edges: ArrayLike, *, include_upper_edge: bool = True) -> NumberSpectrum
```

Bin weighted point diameters on explicit diameter edges.

**Parameters**

- `diameter` (`ArrayLike`) — Particle diameters [m] with shape (..., slots). Zero diameter is allowed for a zero-weight slot. Occupied diameters are positive.
- `number_density` (`ArrayLike`) — Nonnegative weights [m⁻³] matching diameter.
- `diameter_edges` (`ArrayLike`) — Increasing positive bin boundaries [m], shape (bins+1,).
- `include_upper_edge` (`bool`) (default: `True`) — Include the last edge in the last bin (default). Set False when a study counts that edge in an overflow channel.

**Returns**

- (`NumberSpectrum`) — Integrated densities [m⁻³] with shape (..., bins) and densities below
- (`NumberSpectrum`) — and above the covered range with shape (...,). Interior edges belong
- (`NumberSpectrum`) — to the bin on their right.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/number_spectrum.py#L90-L146)

## particula.particles.properties.number_spectrum.get_number_per_decade

*function*

```python
def get_number_per_decade(bin_number_density: ArrayLike, diameter_edges: ArrayLike) -> NDArray[np.float64]
```

Divide integrated bin densities [m⁻³] by their log10 diameter widths.

The shared 1-D edges are in metres. The last input axis enumerates bins.
Returns dN/dlog10D [m⁻³ per decade] with the input density shape.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/number_spectrum.py#L62-L73)

## particula.particles.properties.number_spectrum.get_particle_number_spectrum

*function*

```python
def get_particle_number_spectrum(particles: ParticleData, diameter_edges: ArrayLike) -> NumberSpectrum
```

Bin volume-equivalent diameters of discrete or resolved CPU particles.

Supply diameter edges in metres. Returns one row of integrated number
densities per box [m⁻³]. Resolved counts use the owning box's volume.
Continuous radius PDFs require integration over the radius grid.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/number_spectrum.py#L149-L165)

## particula.particles.properties.number_spectrum.get_rebinned_number_spectrum

*function*

```python
def get_rebinned_number_spectrum(bin_number_density: ArrayLike, source_edges: ArrayLike, target_edges: ArrayLike) -> NumberSpectrum
```

Rebin densities assuming constant dN/dlog10D within each source bin.

Supply nonnegative densities [m⁻³] with the source bin axis last and two
positive diameter edge vectors [m]. Returns integrated densities [m⁻³]
on the target partition and partial or whole bins outside either end.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/number_spectrum.py#L168-L195)
