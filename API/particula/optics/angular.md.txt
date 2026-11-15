# particula.optics.angular

*module*

Angular Mie amplitudes using reusable external field coefficients.

Angular recurrences follow Bohren and Huffman (1983), chapter 4,
https://doi.org/10.1002/9783527618156. Angles are radians; amplitudes use
unnormalized S1/S2 so integral S11 dOmega = pi*x²*Qsca.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/angular.py#L1-L1)

## particula.optics.angular.ScatteringAmplitudes

*class*

```python
class ScatteringAmplitudes(NamedTuple)
```

Bases: `NamedTuple`

Dimensionless complex amplitudes with a final angle axis.

**Attributes**

- `perpendicular` (`NDArray[np.complex128]`) — S1 amplitude perpendicular to the scattering plane.
- `parallel` (`NDArray[np.complex128]`) — S2 amplitude parallel to the scattering plane.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/angular.py#L16-L25)

### particula.optics.angular.ScatteringAmplitudes.parallel

*attribute* · *instance attribute*

```python
parallel: NDArray[np.complex128]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/angular.py#L25-L25)

### particula.optics.angular.ScatteringAmplitudes.perpendicular

*attribute* · *instance attribute*

```python
perpendicular: NDArray[np.complex128]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/angular.py#L24-L24)

## particula.optics.angular.get_scattering_amplitudes

*function*

```python
def get_scattering_amplitudes(coefficients: MieCoefficients, angle: ArrayLike) -> ScatteringAmplitudes
```

Calculate S1 and S2 without repeating the coefficient calculation.

**Parameters**

- `coefficients` (`MieCoefficients`) — Zero-padded results from get_mie_coefficients.
- `angle` (`ArrayLike`) — Scalar or 1D scattering angles [rad] within [0, pi].

**Returns**

- (`ScatteringAmplitudes`) — Complex arrays shaped coefficient batch shape plus (n_angles,).
- (`ScatteringAmplitudes`) — Scalar angle retains a length-one axis. Inputs are unchanged.

**Raises**

- `ValueError` — If angles, coefficient shapes or values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/angular.py#L28-L80)
