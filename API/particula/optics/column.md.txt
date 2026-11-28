# particula.optics.column

*module*

Path-integrated extinction and direct-beam attenuation.

Beer-Lambert transmission describes the unscattered direct beam. It does not
include diffuse multiple scattering or infer path geometry from box volume.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L1-L1)

## particula.optics.column.ColumnOptics

*class*

```python
class ColumnOptics(NamedTuple)
```

Bases: `NamedTuple`

Layer and integrated optical depth and direct-beam transmission.

**Attributes**

- `layer_optical_depth` (`NDArray[np.float64]`) — Extinction times geometric path length [1].
- `optical_depth` (`NDArray[np.float64]`) — Sum of layer depths along the selected layer axis [1].
- `transmission` (`NDArray[np.float64]`) — Unscattered direct-beam fraction exp(-optical_depth).

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L22-L33)

### particula.optics.column.ColumnOptics.layer_optical_depth

*attribute* · *instance attribute*

```python
layer_optical_depth: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L31-L31)

### particula.optics.column.ColumnOptics.optical_depth

*attribute* · *instance attribute*

```python
optical_depth: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L32-L32)

### particula.optics.column.ColumnOptics.transmission

*attribute* · *instance attribute*

```python
transmission: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L33-L33)

## particula.optics.column.SpectralColumn

*class* · *dataclass*

```python
class SpectralColumn
```

Vertical extinction, scattering and phase function of homogeneous layers.

**Attributes**

- `wavelength_m` (`NDArray`) — Strictly increasing positive vacuum wavelengths [m].
- `interface_height_m` (`NDArray`) — Interface heights [m] in strictly descending order, including both the top and bottom of the column.
- `optical_depth` (`NDArray`) — Nonnegative vertical extinction optical depth [dimensionless], shape (layer, wavelength).
- `single_scattering_albedo` (`NDArray`) — Scattering fraction of extinction in [0, 1] with the same axes as optical_depth.
- `phase_moments` (`NDArray`) — Normalized Legendre coefficients of the phase function, shape (layer, wavelength, order). Each lies in [-1, 1]. Order zero equals one wherever scattering depth is positive. Order one is the mean cosine of the scattering angle.
- `sources` (`tuple[str, ...]`) — Provenance of the optical properties and vertical grid.

**Raises**

- `ValueError` — Input arrays are nonfinite, the axes disagree, or a
coordinate or optical property violates its physical domain.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L114-L323)

### particula.optics.column.SpectralColumn.interface_height_m

*attribute* · *instance attribute*

```python
interface_height_m: NDArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L138-L138)

### particula.optics.column.SpectralColumn.optical_depth

*attribute* · *instance attribute*

```python
optical_depth: NDArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L139-L139)

### particula.optics.column.SpectralColumn.phase_moments

*attribute* · *instance attribute*

```python
phase_moments: NDArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L141-L141)

### particula.optics.column.SpectralColumn.single_scattering_albedo

*attribute* · *instance attribute*

```python
single_scattering_albedo: NDArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L140-L140)

### particula.optics.column.SpectralColumn.sources

*attribute* · *class attribute* · *instance attribute*

```python
sources: tuple[str, ...] = ('user layer optics and interface heights')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L142-L142)

### particula.optics.column.SpectralColumn.wavelength_m

*attribute* · *instance attribute*

```python
wavelength_m: NDArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L137-L137)

### particula.optics.column.SpectralColumn.__init__

*method*

```python
def __init__(wavelength_m: NDArray, interface_height_m: NDArray, optical_depth: NDArray, single_scattering_albedo: NDArray, phase_moments: NDArray, sources: tuple[str, ...] = ('user layer optics and interface heights')) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L0-L0)

### particula.optics.column.SpectralColumn.__post_init__

*method*

```python
def __post_init__()
```

Freeze and validate the layer and wavelength axes.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L144-L174)

### particula.optics.column.SpectralColumn.from_population

*method* · *classmethod*

```python
def from_population(wavelength_m, interface_height_m, population, box_indices=None, *, hg_order=32)
```

Convert homogeneous population optics to vertical layer optics.

Extinction times geometrical thickness gives vertical optical depth.
The Henyey–Greenstein closure sets the moment of order l to g**l, where
g is the population asymmetry parameter.

**Parameters**

- `wavelength_m` (`ArrayLike`) — Increasing vacuum wavelengths [m] of the population optics, shape (wavelength,).
- `interface_height_m` (`ArrayLike`) — Strictly descending interface heights [m].
- `population` (`PopulationOptics`) — Bulk optics with extinction [m⁻¹], single scattering albedo and asymmetry (both dimensionless), each with axes (box, wavelength).
- `box_indices` (`ArrayLike | None`) (default: `None`) — Integer box index for each layer, in top-first order. If omitted, each row of population optics supplies one layer.
- `hg_order` (`int`) (default: `32`) — Highest phase moment to retain, an integer from 1 to 255.

**Returns**

- (`SpectralColumn`) — One layer per adjacent pair of interfaces and hg_order + 1 phase moments per layer and wavelength.

**Raises**

- `ValueError` — The grids, extinction, optical properties, box mapping
or requested phase order are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L176-L251)

### particula.optics.column.SpectralColumn.plus

*method*

```python
def plus(other)
```

Combine independent optical constituents in the same layers.

**Parameters**

- `other` (`SpectralColumn`) — Another SpectralColumn on exactly the same wavelength and interface-height grids.

**Returns**

- (`SpectralColumn`) — New column with summed extinction and scattering depths. The phase moments are weighted by scattering optical depth. Missing higher orders contribute zero. Transparent layers receive albedo zero.

**Raises**

- `ValueError` — The two columns have different coordinates.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L276-L323)

### particula.optics.column.SpectralColumn.scaled

*method*

```python
def scaled(factor)
```

Multiply all layer optical depths by a common factor.

**Parameters**

- `factor` (`float`) — Finite nonnegative dimensionless extinction multiplier.

**Returns**

- (`SpectralColumn`) — New column with the same albedos and phase functions. Multiplying by zero gives a transparent column on the original grid.

**Raises**

- `ValueError` — The multiplier is nonscalar, nonfinite or negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L253-L274)

## particula.optics.column.get_column_optics

*function*

```python
def get_column_optics(extinction: ArrayLike, path_length: ArrayLike, *, layer_axis: int = -1) -> ColumnOptics
```

Integrate extinction along explicit path segments.

**Parameters**

- `extinction` (`ArrayLike`) — Nonnegative extinction coefficient [m⁻¹].
- `path_length` (`ArrayLike`) — Nonnegative path through each layer [m], broadcastable with extinction. Supply slant lengths explicitly if applicable.
- `layer_axis` (`int`) (default: `-1`) — Layer axis after broadcasting; scalar input is one layer.

**Returns**

- (`ColumnOptics`) — Dimensionless layer depths, total depth and direct transmission.
- (`ColumnOptics`) — Empty columns have zero depth and unit transmission.

**Raises**

- `ValueError` — If inputs, resulting depths or layer axis are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L36-L76)

## particula.optics.column.get_mass_specific_opacity

*function*

```python
def get_mass_specific_opacity(coefficient: ArrayLike, mass_density: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.bool_]]
```

Divide an optical coefficient by a specified material mass density.

**Parameters**

- `coefficient` (`ArrayLike`) — Nonnegative extinction or absorption [m⁻¹].
- `mass_density` (`ArrayLike`) — Nonnegative density [kg/m³], broadcastable with coefficient. The caller specifies which material inventory it is.

**Returns**

- (`NDArray[np.float64]`) — Mass-specific opacity [m²/kg] and positive-density validity mask.
- (`NDArray[np.bool_]`) — A zero/zero vacuum returns zero and False.

**Raises**

- `ValueError` — If inputs/results are invalid or nonzero optics has zero
density.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L79-L111)
