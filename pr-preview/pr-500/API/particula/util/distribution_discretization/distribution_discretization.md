# Distribution Discretization

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Distribution Discretization

> Auto-generated documentation for [particula.util.distribution_discretization](https://github.com/uncscode/particula/blob/main/particula/util/distribution_discretization.py) module.

## discretize

[Show source in distribution_discretization.py:9](https://github.com/uncscode/particula/blob/main/particula/util/distribution_discretization.py#L9)

discretize the distribution of the particles

#### Arguments

interval    (float) the size interval of the distribution
distype     (str)   the type of distribution, "lognormal" for now
gsigma      (float) geometric standard deviation of distribution
mode        (float) pdf scale (corresponds to mode in lognormal)

#### Signature

```python
def discretize(
    interval=None,
    disttype="lognormal",
    gsigma=in_scalar(1.25).m,
    mode=in_radius(1e-07),
    nparticles=in_scalar(100000.0).m,
    **kwargs
): ...
```
