# Radius Cutoff

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Radius Cutoff

> Auto-generated documentation for [particula.util.radius_cutoff](https://github.com/uncscode/particula/blob/main/particula/util/radius_cutoff.py) module.

## cut_rad

[Show source in radius_cutoff.py:11](https://github.com/uncscode/particula/blob/main/particula/util/radius_cutoff.py#L11)

This routine determins the radius cutoff for the particle distribution

Inputs:
    cutoff  (float) coverage cutoff (default: .9999)
    gsigma  (float) geometric standard deviation (default: 1.25)
    mode    (float) mean radius of the particles (default: 1e-7)

#### Returns

(starting radius, ending radius) float tuple

#### Signature

```python
def cut_rad(
    cutoff=in_scalar(0.9999).m,
    gsigma=in_scalar(1.25).m,
    mode=in_radius(1e-07),
    force_radius_start=None,
    force_radius_end=None,
    **kwargs
): ...
```
