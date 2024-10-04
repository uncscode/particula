# Gibbs

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Activity](./index.md#activity) / Gibbs

> Auto-generated documentation for [particula.activity.gibbs](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs.py) module.

## gibbs_free_engery

[Show source in gibbs.py:6](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs.py#L6)

Calculate the gibbs free energy of the mixture. Ideal and non-ideal.

#### Arguments

- `organic_mole_fraction` *np.array* - A numpy array of organic mole fractions.
- `gibbs_mix` *np.array* - A numpy array of gibbs free energy of mixing.

#### Returns

- `gibbs_ideal` *np.array* - The ideal gibbs free energy of mixing.
- `gibbs_real` *np.array* - The real gibbs free energy of mixing.

#### Signature

```python
def gibbs_free_engery(organic_mole_fraction, gibbs_mix): ...
```
