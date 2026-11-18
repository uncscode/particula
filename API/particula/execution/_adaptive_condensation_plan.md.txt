# particula.execution._adaptive_condensation_plan

*module*

Freeze native material laws for adaptive isothermal device rates.

This profile is separate from the inventory-limited, four-substep prepared
process. It retains every mapped material and the CPU surface phase law.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_adaptive_condensation_plan.py#L1-L1)

## particula.execution._adaptive_condensation_plan.adaptive_condensation_identity

*function*

```python
def adaptive_condensation_identity(strategy, bat_table=None)
```

Identify the material laws and exact resident activity table.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_adaptive_condensation_plan.py#L30-L38)

## particula.execution._adaptive_condensation_plan.plan_adaptive_condensation

*function*

```python
def plan_adaptive_condensation(strategy, aerosol)
```

Validate and freeze constant/Buck, ideal/kappa and phased surfaces.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_adaptive_condensation_plan.py#L41-L157)
