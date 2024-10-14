# Integration

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Lagrangian](./index.md#lagrangian) / Integration

> Auto-generated documentation for [particula.lagrangian.integration](https://github.com/uncscode/particula/blob/main/particula/lagrangian/integration.py) module.

## leapfrog

[Show source in integration.py:7](https://github.com/uncscode/particula/blob/main/particula/lagrangian/integration.py#L7)

Perform a single step of leapfrog integration on the position and velocity
of a particle.

Leapfrog integration is a numerical method used for solving differential
equations typically found in molecular dynamics and astrophysics.
It is symplectic, hence conserves energy over long simulations,
and is known for its simple implementation and stability over
large time steps.

#### Arguments

- position (Tensor): The current position of the particle.
- velocity (Tensor): The current velocity of the particle.
- force (Tensor): The current force acting on the particle.
- mass (float): The mass of the particle.
- time_step (float): The time step for the integration.

#### Returns

- `-` *tuple* - Updated position and velocity of the particle after one time step.

Reference:
- https://en.wikipedia.org/wiki/Leapfrog_integration

#### Signature

```python
def leapfrog(
    position: torch.Tensor,
    velocity: torch.Tensor,
    force: torch.Tensor,
    mass: torch.Tensor,
    time_step: float,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
```
