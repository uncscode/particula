# particula.dynamics.dilution

*module*

Validated, vectorized CPU calculations for chamber dilution.

This module uses chamber volume ``V`` [m³], inlet flow ``Q`` [m³/s], and
dilution coefficient ``alpha`` [s⁻¹], where ``alpha = Q / V``. For
particle-number or gas-mass concentration ``c`` [1/m³ or kg/m³], it evaluates
``dc/dt = -alpha * c`` and the exact finite update
``c_new = c * exp(-alpha * time_step)`` for elapsed time [s].

The helpers validate finite physical domains, use ordinary NumPy broadcasting,
and do not mutate caller-owned arrays. All-scalar inputs return a scalar;
inputs including an array return a broadcast-shape array. The concrete
``dilute_aerosol`` intentionally mutates its ``Aerosol`` argument in place.
It completely validates supported particle and gas concentration state and all
updates before its first write, then restores prior backing state if a later
write fails. ``DilutionStrategy`` is the supported public strategy and
delegates this atomic update to the primitive. The low-level
``dilute_aerosol`` and ``get_dilution_step`` helpers remain concrete-module
APIs and are not exported from ``particula.dynamics``. This module provides no
GPU behavior.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L1-L1)

## particula.dynamics.dilution.DilutionStrategy

*class*

```python
class DilutionStrategy
```

Apply a shared, validated chamber dilution coefficient to an aerosol.

The strategy reports particle-number rates and delegates atomic particle
and gas concentration updates to the concrete dilution primitive. A step
validates its elapsed time and the supported aerosol state before the first
concentration write; malformed state therefore leaves it unchanged. This
strategy is exported from :mod:`particula.dynamics`.

**Parameters**

- `coefficient` (`float | np.number`) — Scalar chamber dilution coefficient [s⁻¹], finite and nonnegative.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L447-L528)

### particula.dynamics.dilution.DilutionStrategy.coefficient

*attribute* · *instance attribute*

```python
coefficient = _validate_nonnegative_scalar(coefficient, 'coefficient')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L468-L470)

### particula.dynamics.dilution.DilutionStrategy.__init__

*method*

```python
def __init__(coefficient: float | np.number)
```

Initialize the strategy with a validated dilution coefficient.

**Parameters**

- `coefficient` (`float | np.number`) — Scalar chamber dilution coefficient [s⁻¹], finite and nonnegative.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L461-L470)

### particula.dynamics.dilution.DilutionStrategy._preflight

*method*

```python
def _preflight(aerosol: Aerosol, time_step: np.float64) -> None
```

Validate supported aerosol state before runnable substeps.

**Parameters**

- `aerosol` (`Aerosol`) — Aerosol whose dilution state is validated without writes.
- `time_step` (`np.float64`) — Validated total elapsed time [s].

**Raises**

- `TypeError` — If required supported aerosol state is missing or not
numeric.
- `ValueError` — If supported aerosol state cannot be safely diluted.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L491-L503)

### particula.dynamics.dilution.DilutionStrategy.rate

*method*

```python
def rate(aerosol: Aerosol) -> float | NDArray[np.float64]
```

Calculate the particle-number dilution rate.

**Parameters**

- `aerosol` (`Aerosol`) — Aerosol providing physical particle-number concentration.

**Returns**

- (`float | NDArray[np.float64]`) — Particle-number concentration rate [1/(m³ s)] with the scalar or
- (`float | NDArray[np.float64]`) — array shape of the physical particle-number concentration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L472-L489)

### particula.dynamics.dilution.DilutionStrategy.step

*method*

```python
def step(aerosol: Aerosol, time_step: float | np.number) -> Aerosol
```

Apply one validated, atomic in-place aerosol dilution step.

Validates the concrete aerosol state and every candidate before its
first write. A preflight failure leaves the aerosol unchanged; a later
write failure restores previously written concentration backing state.

**Parameters**

- `aerosol` (`Aerosol`) — Aerosol whose particle and gas concentrations are diluted.
- `time_step` (`float | np.number`) — Elapsed time [s], finite, nonnegative, and scalar.

**Returns**

- (`Aerosol`) — The identical, mutated aerosol instance.

**Raises**

- `TypeError` — If ``time_step`` is not numeric.
- `ValueError` — If ``time_step`` is nonfinite, negative, or nonscalar,
or supported aerosol state cannot be safely committed.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L505-L528)

## particula.dynamics.dilution._commit_native_dilution

*function*

```python
def _commit_native_dilution(preflight: dict[str, Any]) -> None
```

Publish concentrations, restoring writes on an unexpected error.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L98-L119)

## particula.dynamics.dilution._preflight_concentration

*function*

```python
def _preflight_concentration(candidate: object, source: object, name: str) -> NDArray[np.float64]
```

Validate a concentration candidate before an in-place commit.

**Parameters**

- `candidate` (`object`) — Proposed updated concentration.
- `source` (`object`) — Original concentration that determines the required shape.
- `name` (`str`) — Concentration name used in a validation error message.

**Returns**

- (`NDArray[np.float64]`) — The finite, nonnegative candidate as a float64 array.

**Raises**

- `ValueError` — If the candidate is nonfinite, negative, or has a shape
different from ``source``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L362-L395)

## particula.dynamics.dilution._preflight_native_dilution

*function*

```python
def _preflight_native_dilution(aerosol: Aerosol, coefficient: np.float64, time_step: np.float64, *, writable: bool = True) -> dict[str, Any]
```

Stage native concentration updates without changing caller storage.

Every gas lane is treated equally. Raw resolved counts and PMF/PDF density
storage receive the same decay; volume normalization is only for rates.
Protected-field aliases reject even for zero-work calls.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L33-L95)

## particula.dynamics.dilution._require_finite_result

*function*

```python
def _require_finite_result(result: NDArray[np.float64], name: str) -> NDArray[np.float64]
```

Raise when a computed dilution quantity is nonfinite.

**Parameters**

- `result` (`NDArray[np.float64]`) — Computed dilution quantity.
- `name` (`str`) — Quantity name included in the validation error.

**Returns**

- (`NDArray[np.float64]`) — The validated result.

**Raises**

- `ValueError` — If any computed value is nonfinite.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L141-L159)

## particula.dynamics.dilution._return_scalar_if_appropriate

*function*

```python
def _return_scalar_if_appropriate(result: NDArray[np.float64], *operands: float | NDArray[np.float64]) -> float | NDArray[np.float64]
```

Return a scalar result only when every operand is scalar.

**Parameters**

- `result` (`NDArray[np.float64]`) — NumPy result array from a broadcast calculation.
- `*operands` (`float | NDArray[np.float64]`) (default: `()`) — Original calculation operands used to determine scalar mode.

**Returns**

- (`float | NDArray[np.float64]`) — The scalar value in ``result`` if all operands are scalars; otherwise,
- (`float | NDArray[np.float64]`) — ``result`` unchanged.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L122-L138)

## particula.dynamics.dilution._validate_nonnegative_scalar

*function*

```python
def _validate_nonnegative_scalar(value: object, argument: str) -> np.float64
```

Validate and convert a finite, nonnegative numeric scalar.

Zero-dimensional NumPy numeric arrays are accepted; arrays with other
shapes are rejected so container dilution always has one shared decay.

**Parameters**

- `value` (`object`) — Python or NumPy numeric scalar to validate.
- `argument` (`str`) — Argument name included in validation error messages.

**Returns**

- (`np.float64`) — Validated scalar converted to ``np.float64``.

**Raises**

- `TypeError` — If ``value`` is ``None`` or is not numeric.
- `ValueError` — If ``value`` is nonscalar, nonfinite, or negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L316-L359)

## particula.dynamics.dilution.dilute_aerosol

*function*

```python
def dilute_aerosol(aerosol: Aerosol, coefficient: float | np.number, time_step: float | np.number) -> Aerosol
```

Dilute an aerosol's particle and gas concentrations in place.

Applies ``c_new = c * exp(-coefficient * time_step)`` to physical
particle concentration storage and every gas lane, including inert species.
Before writing, it validates every supported
source, particle-storage representation, volume, and candidate. Thus,
preflight failures are atomic; an unexpected later write failure restores
concentrations already written from snapshots before reraising the
original exception. This low-level helper is deliberately concrete-module
only; construct :class:`DilutionStrategy` for the public API.

**Parameters**

- `aerosol` (`Aerosol`) — Aerosol whose concentrations are updated in place.
- `coefficient` (`float | np.number`) — Scalar chamber dilution coefficient [s⁻¹], finite and nonnegative.
- `time_step` (`float | np.number`) — Scalar elapsed time [s], finite and nonnegative.

**Returns**

- (`Aerosol`) — The identical, mutated ``aerosol`` instance.

**Raises**

- `TypeError` — If a scalar or required supported aerosol concentration or
storage value is not numeric.
- `ValueError` — If either scalar is nonfinite, negative, or nonscalar, or
supported aerosol state or a concentration candidate cannot be
safely committed.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L398-L444)

## particula.dynamics.dilution.get_dilution_rate

*function*

```python
def get_dilution_rate(coefficient: float | NDArray[np.float64], concentration: float | NDArray[np.float64]) -> float | NDArray[np.float64]
```

Calculate the instantaneous dilution rate ``dc/dt = -alpha * c``.

Finite inputs are validated and broadcast with NumPy before multiplication.
The function does not mutate inputs; all-scalar inputs return a scalar, and
any array input returns an array with the broadcast shape.

**Parameters**

- `coefficient` (`float | NDArray[np.float64]`) — Dilution coefficient ``alpha`` [s⁻¹], nonnegative and finite.
- `concentration` (`float | NDArray[np.float64]`) — Particle-number or gas-mass concentration [1/m³ or kg/m³], nonnegative and finite.

**Returns**

- (`float | NDArray[np.float64]`) — Concentration rate of change [1/(m³ s) or kg/(m³ s)] as a scalar for
- (`float | NDArray[np.float64]`) — scalar inputs, otherwise an array with the broadcast shape.

**Raises**

- `TypeError` — If an operand is ``None`` or is not numeric.
- `ValueError` — If an operand is outside its finite physical domain or the
operand shapes cannot be broadcast.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L206-L247)

## particula.dynamics.dilution.get_dilution_step

*function*

```python
def get_dilution_step(coefficient: float | NDArray[np.float64], concentration: float | NDArray[np.float64], time_step: float | NDArray[np.float64]) -> float | NDArray[np.float64]
```

Calculate the exact finite-step concentration, rather than a delta.

This evaluates ``c_new = c * exp(-alpha * time_step)``, not an Euler
approximation, so zero coefficient, concentration, or elapsed time is an
exact no-op. Finite inputs are validated and broadcast with NumPy without
mutating them. All-scalar inputs return a scalar, and any array input
returns an array with the broadcast shape. This helper is deliberately
module-only and is not exported from ``particula.dynamics``.

**Parameters**

- `coefficient` (`float | NDArray[np.float64]`) — Dilution coefficient ``alpha`` [s⁻¹], nonnegative and finite.
- `concentration` (`float | NDArray[np.float64]`) — Initial concentration [1/m³ or kg/m³], nonnegative and finite.
- `time_step` (`float | NDArray[np.float64]`) — Elapsed time [s], nonnegative and finite.

**Returns**

- (`float | NDArray[np.float64]`) — Updated concentration ``c * exp(-alpha * time_step)`` as a scalar for
- (`float | NDArray[np.float64]`) — scalar inputs, otherwise an array with the broadcast shape.

**Raises**

- `TypeError` — If an operand is ``None`` or is not numeric.
- `ValueError` — If an operand is outside its finite physical domain or the
operand shapes cannot be broadcast.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L250-L313)

## particula.dynamics.dilution.get_volume_dilution_coefficient

*function*

```python
def get_volume_dilution_coefficient(volume: float | NDArray[np.float64], input_flow_rate: float | NDArray[np.float64]) -> float | NDArray[np.float64]
```

Calculate the volume dilution coefficient ``alpha = Q / V``.

Finite inputs are validated and broadcast with NumPy before division. The
function does not mutate inputs; all-scalar inputs return a scalar, and any
array input returns an array with the broadcast shape.

**Parameters**

- `volume` (`float | NDArray[np.float64]`) — Chamber volume ``V`` [m³], strictly positive and finite.
- `input_flow_rate` (`float | NDArray[np.float64]`) — Inlet flow rate ``Q`` [m³/s], nonnegative and finite.

**Returns**

- (`float | NDArray[np.float64]`) — Dilution coefficient [s⁻¹] as a scalar for scalar inputs, otherwise an
- (`float | NDArray[np.float64]`) — array with the broadcast shape.

**Raises**

- `TypeError` — If an operand is ``None`` or is not numeric.
- `ValueError` — If an operand is outside its finite physical domain or the
operand shapes cannot be broadcast.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/dilution.py#L162-L203)
