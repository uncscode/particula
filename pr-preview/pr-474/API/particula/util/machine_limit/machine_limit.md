# Machine Limit

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Machine Limit

> Auto-generated documentation for [particula.util.machine_limit](https://github.com/uncscode/particula/blob/main/particula/util/machine_limit.py) module.

## safe_exp

[Show source in machine_limit.py:12](https://github.com/uncscode/particula/blob/main/particula/util/machine_limit.py#L12)

Compute the exponential of each element in the input array, with limits
to prevent overflow based on machine precision.

#### Arguments

- `value` *ArrayLike* - Input array.

#### Returns

- `np.ndarray` - Exponential of the input array with overflow protection.

#### Signature

```python
def safe_exp(value: ArrayLike) -> np.ndarray: ...
```



## safe_log

[Show source in machine_limit.py:28](https://github.com/uncscode/particula/blob/main/particula/util/machine_limit.py#L28)

Compute the natural logarithm of each element in the input array, with
limits to prevent underflow based on machine precision.

#### Arguments

- `value` *ArrayLike* - Input array.

#### Returns

- `np.ndarray` - Natural logarithm of the input array with underflow
protection.

#### Signature

```python
def safe_log(value: ArrayLike) -> np.ndarray: ...
```



## safe_log10

[Show source in machine_limit.py:45](https://github.com/uncscode/particula/blob/main/particula/util/machine_limit.py#L45)

Compute the base 10 logarithm of each element in the input array, with
limits to prevent underflow based on machine precision.

#### Arguments

- `value` *ArrayLike* - Input array.

#### Returns

- `np.ndarray` - Base 10 logarithm of the input array with underflow
protection.

#### Signature

```python
def safe_log10(value: ArrayLike) -> np.ndarray: ...
```
