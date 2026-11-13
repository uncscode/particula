# particula.particles.properties.sort_bins

*module*

Sort bins of distribution representation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/sort_bins.py#L1-L1)

## particula.particles.properties.sort_bins.get_sorted_bins_by_radius

*function*

```python
def get_sorted_bins_by_radius(radius: NDArray[np.float64], distribution: NDArray[np.float64], concentration: NDArray[np.float64], charge: Union[NDArray[np.float64], float]) -> tuple[NDArray[np.float64], NDArray[np.float64], Union[NDArray[np.float64], float]]
```

Ensure distribution bins are sorted by increasing radius.

**Parameters**

- `radius` (`NDArray[np.float64]`) — The radii of the particles.
- `distribution` (`NDArray[np.float64]`) — The distribution of particle sizes or masses.
- `concentration` (`NDArray[np.float64]`) — The concentration of each particle size or mass.
- `charge` (`Union[NDArray[np.float64], float]`) — (Optional) charge per particle; scalar or NumPy array.

**Returns**

- `distribution` (`NDArray[np.float64]`) — The sorted distribution of particle sizes or masses.
- `concentration` (`NDArray[np.float64]`) — The sorted concentration of each particle size/mass.
- `charge` (`Union[NDArray[np.float64], float]`) — The sorted charge of each particle size/mass.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/sort_bins.py#L9-L43)
