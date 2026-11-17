# particula.execution._condensation_jacobian

*module*

Sparse finite differences for local particles coupled through box gas.

A particle rate depends on that particle's composition and its box gas, not
another particle's composition. Gas derivatives follow from the same weighted
mass exchange. Coloring particle columns by material therefore needs only two
rate evaluations per material, independently of particle count. Gas columns
remain explicit finite differences. Coordinates and derivatives are scaled in
exactly the same way as CondensationODE's dense reference Jacobian.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_jacobian.py#L1-L1)

## particula.execution._condensation_jacobian.coordinate_steps

*function*

```python
def coordinate_steps(owner, values)
```

Scale particle differences by current total mass, including dry cores.

Initial wet coordinates can exceed dry equilibrium mass by many orders of
magnitude. A fixed perturbation in those coordinates then probes a different
physical regime. Use the current slot mass as the composition scale, also
for a currently empty material. Gas dependence is linear at fixed particles.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_jacobian.py#L32-L47)

## particula.execution._condensation_jacobian.sparse_jacobian

*function*

```python
def sparse_jacobian(owner, time, values)
```

Differentiate mapped continuous rates without a dense particle matrix.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_jacobian.py#L50-L114)

## particula.execution._condensation_jacobian.supports_local_rates

*function*

```python
def supports_local_rates(strategy)
```

Restrict coloring to known particle-local activity/surface formulas.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_jacobian.py#L15-L29)
