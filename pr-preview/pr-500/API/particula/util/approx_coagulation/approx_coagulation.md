# Approx Coagulation

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Approx Coagulation

> Auto-generated documentation for [particula.util.approx_coagulation](https://github.com/uncscode/particula/blob/main/particula/util/approx_coagulation.py) module.

## approx_coag_less

[Show source in approx_coagulation.py:10](https://github.com/uncscode/particula/blob/main/particula/util/approx_coagulation.py#L10)

 dy2007 approx.
Dimensionless particle--particle coagulation kernel.

gh2012:
https://journals.aps.org/pre/abstract/10.1103/PhysRevE.85.026410

gk2008:
https://journals.aps.org/pre/abstract/10.1103/PhysRevE.78.046402

dy2007:
https://aip.scitation.org/doi/10.1063/1.2713719

cg2019:
https://www.tandfonline.com/doi/suppl/10.1080/02786826.2019.1614522

todolater:
- quick fixes for corner cases
- examine better solutions?

#### Signature

```python
def approx_coag_less(diff_knu=None, cpr=None, approx="hardsphere"): ...
```
