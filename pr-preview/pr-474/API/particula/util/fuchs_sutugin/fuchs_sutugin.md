# Fuchs Sutugin

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Fuchs Sutugin

> Auto-generated documentation for [particula.util.fuchs_sutugin](https://github.com/uncscode/particula/blob/main/particula/util/fuchs_sutugin.py) module.

## fsc

[Show source in fuchs_sutugin.py:9](https://github.com/uncscode/particula/blob/main/particula/util/fuchs_sutugin.py#L9)

Returns the Fuchs-Sutugin model transition regime correction.

#### Arguments

knu     (float)  [ ] (default: util)
alpha   (float)  [ ] (default: 1)

#### Returns

(float)  [ ]

#### Notes

knu can be calculated using knu(**kwargs);
refer to particula.util.knudsen_number.knu for more info.

#### Signature

```python
def fsc(knu_val=None, alpha=1, **kwargs): ...
```
