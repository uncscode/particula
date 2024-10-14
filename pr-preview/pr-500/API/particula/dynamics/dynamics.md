# Dynamics

[Particula Index](../README.md#particula-index) / [Particula](./index.md#particula) / Dynamics

> Auto-generated documentation for [particula.dynamics](https://github.com/uncscode/particula/blob/main/particula/dynamics.py) module.

## Solver

[Show source in dynamics.py:9](https://github.com/uncscode/particula/blob/main/particula/dynamics.py#L9)

dynamic solver

#### Signature

```python
class Solver(Rates):
    def __init__(
        self,
        time_span=None,
        do_coagulation=True,
        do_condensation=True,
        do_nucleation=True,
        do_dilution=False,
        do_wall_loss=False,
        **kwargs
    ): ...
```

#### See also

- [Rates](./rates.md#rates)

### Solver()._ode_func

[Show source in dynamics.py:35](https://github.com/uncscode/particula/blob/main/particula/dynamics.py#L35)

ode_func

#### Signature

```python
def _ode_func(self, _nums, _): ...
```

### Solver().solution

[Show source in dynamics.py:53](https://github.com/uncscode/particula/blob/main/particula/dynamics.py#L53)

solve the equation

#### Signature

```python
def solution(self, method="odeint", **kwargs_ode): ...
```
