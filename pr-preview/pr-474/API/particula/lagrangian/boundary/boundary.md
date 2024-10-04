# Boundary

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Lagrangian](./index.md#lagrangian) / Boundary

> Auto-generated documentation for [particula.lagrangian.boundary](https://github.com/uncscode/particula/blob/main/particula/lagrangian/boundary.py) module.

## wrapped_cube

[Show source in boundary.py:6](https://github.com/uncscode/particula/blob/main/particula/lagrangian/boundary.py#L6)

Apply cubic boundary conditions with wrap-around, to the position tensor.

This function modifies positions that exceed the cubic domain side,
wrapping them around to the opposite side of the domain. It handles both
positive and negative overflows. The center of the cube is assumed to be
at zero. If a particle is way outside the cube, it is wrapped around to
the opposite side of the cube.

#### Arguments

- position (torch.Tensor): A tensor representing positions that might
    exceed the domain boundaries. [3, num_particles]
- cube_side (float): The cube side length of the domain.

#### Returns

- `-` *torch.Tensor* - The modified position tensor with boundary conditions
    applied.

#### Examples

position = torch.tensor([...])  # Position tensor
cube_side = 10.0  # Define the domain
wrapped_position = boundary.wrapped_cube(position,
    cube_side)

#### Signature

```python
def wrapped_cube(position: torch.Tensor, cube_side: float) -> torch.Tensor: ...
```
