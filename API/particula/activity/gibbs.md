# particula.activity.gibbs

*module*

Calculates the Gibbs free energy of mixing for a binary solution.

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs.py#L1-L1)

## particula.activity.gibbs.gibbs_free_energy

*function*

```python
def gibbs_free_energy(organic_mole_fraction: NDArray[np.float64], gibbs_mix: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]
```

Compute ideal and real Gibbs free energy of mixing.

**Parameters**

- `organic_mole_fraction` (`NDArray[np.float64]`) — Mole fraction of the organic component. Values should be between 0 and 1.
- `gibbs_mix` (`NDArray[np.float64]`) — Gibbs free energy of mixing (non-ideal contribution).

**Returns**

- (`Tuple[NDArray[np.float64], NDArray[np.float64]]`) — Tuple containing: gibbs_ideal: Ideal Gibbs free energy of mixing. gibbs_real: Real Gibbs free energy of mixing (ideal + mix).

**Examples**

```pycon
>>> import numpy as np
>>> from particula.activity.gibbs import gibbs_free_energy
>>> x_org = np.array([0.2, 0.5, 0.8])
>>> g_mix = np.array([0.1, 0.2, 0.1])
>>> g_ideal, g_real = gibbs_free_energy(x_org, g_mix)
>>> g_ideal.shape, g_real.shape
((3,), (3,))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs.py#L12-L41)

## particula.activity.gibbs.gibbs_free_engery

*function*

```python
def gibbs_free_engery(organic_mole_fraction: NDArray[np.float64], gibbs_mix: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]
```

Deprecated alias for gibbs_free_energy.

.. deprecated:: 0.3.0
    Use :func:`gibbs_free_energy` instead. This alias will be removed
    in a future version.

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs.py#L44-L60)
