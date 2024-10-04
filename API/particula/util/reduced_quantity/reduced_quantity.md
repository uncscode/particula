# Reduced Quantity

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Reduced Quantity

> Auto-generated documentation for [particula.util.reduced_quantity](https://github.com/uncscode/particula/blob/main/particula/util/reduced_quantity.py) module.

## reduced_quantity

[Show source in reduced_quantity.py:17](https://github.com/uncscode/particula/blob/main/particula/util/reduced_quantity.py#L17)

Returns the reduced mass of two particles.

#### Examples

```
>>> reduced_quantity(1*u.kg, 1*u.kg)
<Quantity(0.5, 'kilogram')>
>>> reduced_quantity(1*u.kg, 20*u.kg).m
0.9523809523809523
>>> reduced_quantity(1, 200)
0.9950248756218906
>>> reduced_quantity([1, 2, 3], 200)
array([0.99502488, 1.98019802, 2.95566502])
>>> reduced_quantity([1, 2], [200, 300])
array([0.99502488, 1.98675497])
```

#### Arguments

a_quantity  (float)  [arbitrary units]
b_quantity  (float)  [arbitrary units]

#### Returns

(float)  [arbitrary units]

A reduced quantity is an "effective inertial" quantity,
allowing two-body problems to be solved as one-body problems.

#### Signature

```python
def reduced_quantity(a_quantity, b_quantity): ...
```



## reduced_self_broadcast

[Show source in reduced_quantity.py:128](https://github.com/uncscode/particula/blob/main/particula/util/reduced_quantity.py#L128)

Returns the reduced value of an array with itself, broadcasting the
array into a matrix and calculating the reduced value of each element pair.
reduced_value = alpha_matrix * alpha_matrix_Transpose
                / (alpha_matrix + alpha_matrix_Transpose)

#### Arguments

- `-` *alpha_array* - The array to be broadcast and reduced.

#### Returns

-------
- A square matrix of the reduced values.

#### Signature

```python
def reduced_self_broadcast(alpha_array: NDArray[np.float64]) -> NDArray[np.float64]: ...
```



## reduced_value

[Show source in reduced_quantity.py:89](https://github.com/uncscode/particula/blob/main/particula/util/reduced_quantity.py#L89)

Returns the reduced value of two parameters, calculated as:
reduced_value = alpha * beta / (alpha + beta)

This formula calculates an "effective inertial" quantity,
allowing two-body problems to be solved as if they were one-body problems.

#### Arguments

- `-` *alpha* - The first parameter (scalar or array).
- `-` *beta* - The second parameter (scalar or array).

#### Returns

-------
- A value or array of the same dimension as the input parameters. Returns
  zero where alpha + beta equals zero to handle division by zero
  gracefully.

#### Raises

- `-` *ValueError* - If alpha and beta are arrays and their shapes do not match.

#### Signature

```python
def reduced_value(
    alpha: Union[float, NDArray[np.float64]], beta: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```
