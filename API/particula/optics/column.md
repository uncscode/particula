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

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L15-L26)

### particula.optics.column.ColumnOptics.layer_optical_depth

*attribute* · *instance attribute*

```python
layer_optical_depth: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L24-L24)

### particula.optics.column.ColumnOptics.optical_depth

*attribute* · *instance attribute*

```python
optical_depth: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L25-L25)

### particula.optics.column.ColumnOptics.transmission

*attribute* · *instance attribute*

```python
transmission: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L26-L26)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L29-L69)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L72-L104)
