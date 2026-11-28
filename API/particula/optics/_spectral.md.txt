# particula.optics._spectral

*module*

Shared coordinates for spectral radiation and its boundary conditions.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/_spectral.py#L1-L1)

## particula.optics._spectral.finite_array

*function*

```python
def finite_array(values, name)
```

Copy a real-valued physical field into a read-only float64 array.

**Parameters**

- `values` (`ArrayLike`) — Array-like field. Its axes and units are preserved.
- `name` (`str`) — Field name used in validation errors.

**Returns**

- (`ndarray`) — Independent, read-only NumPy array with the input shape.

**Raises**

- `ValueError` — A value is complex, nonfinite or cannot convert to float.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/_spectral.py#L6-L25)

## particula.optics._spectral.interfaces

*function*

```python
def interfaces(values)
```

Validate the top-first vertical grid used by optical layers.

**Parameters**

- `values` (`ArrayLike`) — One-dimensional sequence of interface altitudes [m].

**Returns**

- (`ndarray`) — Read-only strictly descending altitudes [m]. Adjacent pairs bound homogeneous layers.

**Raises**

- `ValueError` — The input is invalid or has fewer than two interfaces.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/_spectral.py#L52-L69)

## particula.optics._spectral.require_same_grid

*function*

```python
def require_same_grid(left, right, name)
```

Require identical coordinate values before combining spectral fields.

**Parameters**

- `left` (`ArrayLike | None`) — First coordinate, or None if absent.
- `right` (`ArrayLike | None`) — Second coordinate in the same units and axis order.
- `name` (`str`) — Coordinate name used in validation errors.

**Raises**

- `ValueError` — The shapes or values differ.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/_spectral.py#L91-L104)

## particula.optics._spectral.spectral_values

*function*

```python
def spectral_values(values, wave, name)
```

Broadcast a finite spectral field onto a validated wavelength grid.

**Parameters**

- `values` (`ArrayLike`) — Scalar or array broadcastable to the one-dimensional grid.
- `wave` (`ndarray`) — Wavelength coordinate with the required output shape.
- `name` (`str`) — Name used in validation errors.

**Returns**

- (`ndarray`) — Read-only float64 values with the shape of wave and units of values.

**Raises**

- `ValueError` — The field is invalid or cannot broadcast to the grid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/_spectral.py#L72-L88)

## particula.optics._spectral.wavelengths

*function*

```python
def wavelengths(values)
```

Validate the vacuum wavelength coordinate shared by shortwave fields.

**Parameters**

- `values` (`ArrayLike`) — One-dimensional sequence of positive wavelengths [m].

**Returns**

- (`ndarray`) — Read-only wavelength array [m] in strictly increasing order.

**Raises**

- `ValueError` — The input is empty, invalid or out of order.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/_spectral.py#L28-L49)
