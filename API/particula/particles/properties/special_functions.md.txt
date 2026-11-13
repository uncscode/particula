# particula.particles.properties.special_functions

*module*

Special non-standard functions for aerosol properties.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/special_functions.py#L1-L1)

## particula.particles.properties.special_functions.get_debye_function

*function*

```python
def get_debye_function(variable: Union[float, NDArray[np.float64]], integration_points: int = 1000, n: int = 1) -> Union[float, NDArray[np.float64]]
```

Calculate the generalized Debye function for a given input.

The Debye function can be expressed as follows:

- Dₙ(x) = (n / xⁿ) ∫[tⁿ / (exp(t) - 1)] dt  from t = 0 to x
    - x is a dimensionless variable.
    - n is the exponent (default is 1).

**Parameters**

- `variable` (`Union[float, NDArray[np.float64]]`) — Upper limit of integration; can be float or NDArray.
- `integration_points` (`int`) (default: `1000`) — Number of points for numerical integration (default 1000).
- `n` (`int`) (default: `1`) — Exponent in the Debye function formula (default 1).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Debye function value(s). If the input is a float, returns a float. If the input is an array, returns an array of the same shape.

**Examples**

``` py title="Debye function with n=1 for a single float value"
import particula as par
par.particles.get_debye_function(1.0)
# Output: 0.7765038970390566
```

``` py title="Debye function with n=2 for a single float value"
import particula as par
par.particles.get_debye_function(1.0, n=2)
# Output: 0.6007582206816492
```

``` py title="Debye function with n=1 for a numpy array"
import particula as par
par.particles.get_debye_function(np.array([1.0, 2.0, 3.0]))
# Output: [0.84140566 0.42278434 0.28784241]
```

> **References**
>
> - [Debye function](https://en.wikipedia.org/wiki/Debye_function)
> - [Wolfram MathWorld: Debye Functions](https://mathworld.wolfram.com/DebyeFunctions.html)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/special_functions.py#L12-L71)
