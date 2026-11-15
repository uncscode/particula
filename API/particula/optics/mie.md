# particula.optics.mie

*module*

Independent homogeneous-sphere Mie calculations in SI units.

The coefficient equations follow Bohren and Huffman (1983), chapter 4,
https://doi.org/10.1002/9783527618156. Logarithmic derivatives use a bounded
continued fraction followed by downward recurrence; external regular Bessel
functions use a convergent power series for small arguments.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L1-L1)

## particula.optics.mie.MieCoefficients

*class*

```python
class MieCoefficients(NamedTuple)
```

Bases: `NamedTuple`

Complex coefficients, zero padded along the final order axis.

**Attributes**

- `electric` (`NDArray[np.complex128]`) — Electric coefficients a_n, starting at order one.
- `magnetic` (`NDArray[np.complex128]`) — Magnetic coefficients b_n, starting at order one.
- `term_count` (`NDArray[np.int64]`) — Valid number of coefficients for each broadcast case.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L15-L26)

### particula.optics.mie.MieCoefficients.electric

*attribute* · *instance attribute*

```python
electric: NDArray[np.complex128]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L24-L24)

### particula.optics.mie.MieCoefficients.magnetic

*attribute* · *instance attribute*

```python
magnetic: NDArray[np.complex128]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L25-L25)

### particula.optics.mie.MieCoefficients.term_count

*attribute* · *instance attribute*

```python
term_count: NDArray[np.int64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L26-L26)

## particula.optics.mie.MieEfficiencies

*class*

```python
class MieEfficiencies(NamedTuple)
```

Bases: `NamedTuple`

Dimensionless efficiencies and asymmetry for broadcast input cases.

**Attributes**

- `extinction` (`NDArray[np.float64]`) — Total extinction efficiency.
- `scattering` (`NDArray[np.float64]`) — Scattering efficiency.
- `absorption` (`NDArray[np.float64]`) — Absorption efficiency.
- `asymmetry` (`NDArray[np.float64]`) — Scattering-weighted cosine, zero when undefined.
- `backscatter` (`NDArray[np.float64]`) — Radar-convention backscatter efficiency.
- `radiation_pressure` (`NDArray[np.float64]`) — Extinction minus asymmetry times scattering.
- `scattering_valid` (`NDArray[np.bool_]`) — Whether asymmetry is defined.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L29-L48)

### particula.optics.mie.MieEfficiencies.absorption

*attribute* · *instance attribute*

```python
absorption: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L44-L44)

### particula.optics.mie.MieEfficiencies.asymmetry

*attribute* · *instance attribute*

```python
asymmetry: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L45-L45)

### particula.optics.mie.MieEfficiencies.backscatter

*attribute* · *instance attribute*

```python
backscatter: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L46-L46)

### particula.optics.mie.MieEfficiencies.extinction

*attribute* · *instance attribute*

```python
extinction: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L42-L42)

### particula.optics.mie.MieEfficiencies.radiation_pressure

*attribute* · *instance attribute*

```python
radiation_pressure: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L47-L47)

### particula.optics.mie.MieEfficiencies.scattering

*attribute* · *instance attribute*

```python
scattering: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L43-L43)

### particula.optics.mie.MieEfficiencies.scattering_valid

*attribute* · *instance attribute*

```python
scattering_valid: NDArray[np.bool_]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L48-L48)

## particula.optics.mie._coefficients

*function*

```python
def _coefficients(m: complex, x: float, count: int)
```

Calculate one admitted case without allocating broadcast outputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L88-L117)

## particula.optics.mie._log_derivative

*function*

```python
def _log_derivative(z: complex, order: int) -> complex
```

Evaluate D_order(z) using the modified Lentz continued fraction.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L51-L70)

## particula.optics.mie._regular_small

*function*

```python
def _regular_small(x: float, order: int) -> float
```

Evaluate x*j_order(x) by its convergent small-argument series.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L73-L85)

## particula.optics.mie.get_mie_coefficients

*function*

```python
def get_mie_coefficients(relative_refractive_index: ArrayLike, size_parameter: ArrayLike) -> MieCoefficients
```

Calculate external coefficients for nonmagnetic homogeneous spheres.

**Parameters**

- `relative_refractive_index` (`ArrayLike`) — Particle/medium index n+ik, n>0, k>=0.
- `size_parameter` (`ArrayLike`) — Dimensionless 2*pi*r*n_medium/lambda_vacuum, 0..10000.

**Returns**

- (`MieCoefficients`) — Broadcast coefficients with a final, zero-padded order axis and
- (`MieCoefficients`) — per-case valid term counts. Inputs are not mutated.

**Raises**

- `ValueError` — If physical inputs are nonfinite or outside the domain.
- `RuntimeError` — If the numerical recurrence fails.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L120-L173)

## particula.optics.mie.get_mie_efficiencies

*function*

```python
def get_mie_efficiencies(radius: ArrayLike, wavelength: ArrayLike, refractive_index: ArrayLike, medium_index: ArrayLike = 1.0) -> MieEfficiencies
```

Calculate efficiencies from SI radius and vacuum wavelength.

**Parameters**

- `radius` (`ArrayLike`) — Nonnegative sphere radius [m].
- `wavelength` (`ArrayLike`) — Positive vacuum wavelength [m].
- `refractive_index` (`ArrayLike`) — Absolute particle refractive index n+ik.
- `medium_index` (`ArrayLike`) (default: `1.0`) — Positive real nonabsorbing medium index, default 1.

**Returns**

- (`MieEfficiencies`) — Named dimensionless arrays with the broadcast input shape. Zero
- (`MieEfficiencies`) — scattering gives asymmetry zero and scattering_valid False.

**Raises**

- `ValueError` — If inputs violate the physical domain or x exceeds 10000.
- `RuntimeError` — If arithmetic is unresolved, coefficient calculation
fails or efficiencies violate energy/physical bounds beyond
64 float64 eps of roundoff. Negative absorption within that
roundoff is set to zero with extinction adjusted to scattering.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L176-L282)
