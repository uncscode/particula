# particula.util.arbitrary_round

*module*

Rounding function that allows for arbitrary bases and rounding modes.

To be removed, likely particula_beta only. -kyle

[View source](https://github.com/uncscode/particula/blob/main/particula/util/arbitrary_round.py#L1-L1)

## particula.util.arbitrary_round.get_arbitrary_round

*function*

```python
def get_arbitrary_round(values: Union[float, list[float], np.ndarray], base: Union[float, np.float64] = 1.0, mode: str = 'round', nonzero_edge: bool = False) -> Union[float, NDArray[np.float64]]
```

Round values to the nearest multiple of a specified base.

The function supports "round", "floor", or "ceil" modes, and can retain
original nonzero values if rounding returns zero.

**Parameters**

- `values` (`Union[float, list[float], np.ndarray]`) — The values to be rounded.
- `base` (`Union[float, np.float64]`) (default: `1.0`) — Positive float indicating the rounding interval.
- `mode` (`str`) (default: `'round'`) — Rounding mode, one of ['round', 'floor', 'ceil'].
- `nonzero_edge` (`bool`) (default: `False`) — If True, zeros after rounding are replaced with the original values.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The input values rounded according to the specified base and mode.

**Examples**

``` py title="Example Usage"
import numpy as np
import particula as par

arr = np.array([1.2, 2.5, 3.7, 4.0])
print(par.get_arbitrary_round(arr, base=1.0, mode='round'))
# Output: [1.  2.  4.  4.]

print(par.get_arbitrary_round(arr, base=0.5, mode='floor'))
# Output: [1.  2.  3.5 4. ]

print(par.get_arbitrary_round(2.5, base=1.0, mode='round'))
# Output: 2.0
```

> **References**
>
> - "Rounding," Python Documentation, docs.python.org.
> - "NumPy Rounding," NumPy Documentation, NumPy.org.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/arbitrary_round.py#L14-L79)
