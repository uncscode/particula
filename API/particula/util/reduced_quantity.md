# particula.util.reduced_quantity

*module*

This module provides functions to calculate a reduced quantity between
parameters or across an array, useful in multi-body or multi-parameter
problems.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/reduced_quantity.py#L1-L1)

## particula.util.reduced_quantity.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/util/reduced_quantity.py#L12-L12)

## particula.util.reduced_quantity.get_reduced_self_broadcast

*function*

```python
def get_reduced_self_broadcast(alpha_array: NDArray[np.float64]) -> NDArray[np.float64]
```

Return a square matrix of pairwise reduced values using a single array.

Each element is calculated by broadcasting the array with its transpose:
- r_ij = (α_i × α_j) / (α_i + α_j),
    - r_ij is the reduced quantity between α_i and α_j.

**Parameters**

- `alpha_array` (`NDArray[np.float64]`) — A 1D array for pairwise reduced value calculations.

**Returns**

- (`NDArray[np.float64]`) — A 2D square matrix of pairwise reduced values.

**Examples**

``` py title="Example"
from particula.util.reduced_quantity import get_reduced_self_broadcast
import numpy as np

arr = np.array([1.0, 2.0, 3.0])
print(get_reduced_self_broadcast(arr))
# Output: [[0.5       0.6666667 0.75     ]
#          [0.6666667 1.        1.2      ]
#          [0.75      1.2       1.5      ]]
```

> **References**
>
> - [Reduced Mass, Wikipedia](https://en.wikipedia.org/wiki/Reduced_mass)

[View source](https://github.com/uncscode/particula/blob/main/particula/util/reduced_quantity.py#L70-L111)

## particula.util.reduced_quantity.get_reduced_value

*function*

```python
def get_reduced_value(alpha: Union[float, NDArray[np.float64]], beta: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Return the reduced value of two parameters.

The reduced value is computed using:
- r = (α × β) / (α + β),
    - r is the reduced quantity,
    - α, β are the input parameters.

**Parameters**

- `alpha` (`Union[float, NDArray[np.float64]]`) — The first parameter (scalar or array).
- `beta` (`Union[float, NDArray[np.float64]]`) — The second parameter (scalar or array).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The element-wise reduced quantity, zero if (α+β)=0.

**Raises**

- `ValueError` — If arrays have incompatible shapes.

**Examples**

``` py title="Example"
from particula.util.reduced_quantity import get_reduced_value
import numpy as np

print(get_reduced_value(3.0, 6.0))
# Output: 2.0

arrA = np.array([1.0, 2.0, 3.0])
arrB = np.array([2.0, 5.0, 10.0])
print(get_reduced_value(arrA, arrB))
# Output: [0.666..., 1.428..., 2.142...]
```

> **References**
>
> - [Reduced Mass, Wikipedia](https://en.wikipedia.org/wiki/Reduced_mass)

[View source](https://github.com/uncscode/particula/blob/main/particula/util/reduced_quantity.py#L15-L67)
