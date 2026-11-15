# particula.optics.polarization

*module*

Scattering-plane Mueller matrices and Stokes operations.

Use (parallel, perpendicular) electric-field components with
Q=|Eparallel|²-|Eperpendicular|², U=2 Re(Eparallel Eperpendicular*), and
V=-2 Im(Eparallel Eperpendicular*). Sphere scattering has Jones diagonal
(S2, S1). Equations follow Bohren and Huffman (1983), chapter 3,
https://doi.org/10.1002/9783527618156.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/polarization.py#L1-L1)

## particula.optics.polarization.apply_mueller_matrix

*function*

```python
def apply_mueller_matrix(matrix: ArrayLike, stokes: ArrayLike, input_rotation: ArrayLike = 0.0, output_rotation: ArrayLike = 0.0) -> NDArray[np.float64]
```

Apply R(output_rotation) M R(input_rotation) to a Stokes vector.

**Parameters**

- `matrix` (`ArrayLike`) — Finite (..., 4, 4) Mueller array; units set output units.
- `stokes` (`ArrayLike`) — Finite (..., 4) I,Q,U,V input, broadcastable with matrix.
- `input_rotation` (`ArrayLike`) (default: `0.0`) — Rotation [rad] into the matrix's incident basis.
- `output_rotation` (`ArrayLike`) (default: `0.0`) — Rotation [rad] from its outgoing basis.

**Returns**

- (`NDArray[np.float64]`) — New output Stokes vectors, without intensity normalization or
- (`NDArray[np.float64]`) — propagation-distance factors. Signed linear combinations are allowed.

**Raises**

- `ValueError` — If matrix/vector shapes or finite values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/polarization.py#L78-L106)

## particula.optics.polarization.get_mueller_matrix

*function*

```python
def get_mueller_matrix(perpendicular: ArrayLike, parallel: ArrayLike) -> NDArray[np.float64]
```

Construct the unnormalized sphere Mueller matrix from S1 and S2.

**Parameters**

- `perpendicular` (`ArrayLike`) — Complex perpendicular amplitude S1, dimensionless.
- `parallel` (`ArrayLike`) — Complex parallel amplitude S2, broadcastable with S1.

**Returns**

- (`NDArray[np.float64]`) — Real matrix with broadcast shape plus (4, 4), in the scattering
- (`NDArray[np.float64]`) — plane. Divide by medium wavenumber squared for m²/sr.

**Raises**

- `ValueError` — If either amplitude is nonfinite.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/polarization.py#L14-L45)

## particula.optics.polarization.rotate_stokes

*function*

```python
def rotate_stokes(stokes: ArrayLike, angle: ArrayLike) -> NDArray[np.float64]
```

Express a Stokes vector in axes rotated toward the perpendicular axis.

**Parameters**

- `stokes` (`ArrayLike`) — Finite (..., 4) array ordered I,Q,U,V, in caller units.
- `angle` (`ArrayLike`) — Rotation [rad], broadcastable with the leading Stokes shape. Positive rotation uses Eparallel'=cos(angle)*Eparallel + sin(angle)*Eperpendicular.

**Returns**

- (`NDArray[np.float64]`) — New vectors with Q'=cos(2a)Q+sin(2a)U, U'=-sin(2a)Q+cos(2a)U.
- (`NDArray[np.float64]`) — I and V are unchanged. This operation does not normalize intensity.

**Raises**

- `ValueError` — If shapes or finite values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/polarization.py#L48-L75)
