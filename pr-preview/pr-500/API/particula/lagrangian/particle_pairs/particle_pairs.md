# Particle Pairs

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Lagrangian](./index.md#lagrangian) / Particle Pairs

> Auto-generated documentation for [particula.lagrangian.particle_pairs](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_pairs.py) module.

## calculate_pairwise_distance

[Show source in particle_pairs.py:48](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_pairs.py#L48)

need to test this:

Calculate the pairwise Euclidean distances between points in a given
position tensor.

This function computes the pairwise distances between points represented
in the input tensor. Each row of the input tensor is considered a point in
n-dimensional space.

#### Arguments

- `position` *torch.Tensor* - A 2D tensor of shape [n_dimensions, n_points]

#### Returns

- `torch.Tensor` - A 2D tensor of shape [n_points, n_points] containing the
pairwise Euclidean distances between each pair of points.
The element at [i, j] in the output tensor represents the distance
between the i-th and j-th points in the input tensor.

#### Examples

position = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# Output will be a 3x3 tensor with the pairwise distances between these
3 points.

#### Signature

```python
def calculate_pairwise_distance(position: torch.Tensor) -> torch.Tensor: ...
```



## full_sweep_and_prune

[Show source in particle_pairs.py:159](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_pairs.py#L159)

Sweep and prune algorithm for collision detection along all three axes
(x, y, z). This function identifies pairs of particles that are close
enough to potentially collide in 3D space.

#### Arguments

- `position` *torch.Tensor* - The 2D tensor of particle positions,
    where each row represents an axis (x, y, z).
- `radius` *torch.Tensor* - The radius of particles.

#### Returns

- `torch.Tensor` - A tensor containing pairs of indices of potentially
    colliding particles.

#### Signature

```python
def full_sweep_and_prune(
    position: torch.Tensor, radius: torch.Tensor
) -> torch.Tensor: ...
```



## full_sweep_and_prune_simplified

[Show source in particle_pairs.py:240](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_pairs.py#L240)

A simplified version of the full sweep and prune algorithm for collision
written above, it is not working yet. there is an error in the update of
the indices in the y and z axis.

Sweep and prune algorithm for collision detection along all three axes
(x, y, z). This function identifies pairs of particles that are close
enough to potentially collide in 3D space.

#### Arguments

- `position` *torch.Tensor* - The 2D tensor of particle positions,
    where each row represents an axis (x, y, z).
- `radius` *torch.Tensor* - The radius of particles.

#### Returns

- `torch.Tensor` - A tensor containing pairs of indices of potentially
    colliding particles.

#### Signature

```python
def full_sweep_and_prune_simplified(
    position: torch.Tensor, radius: torch.Tensor, working_yet: bool = False
) -> torch.Tensor: ...
```



## remove_duplicates

[Show source in particle_pairs.py:7](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_pairs.py#L7)

Removes duplicate entries from a specified column in a tensor of index
pairs.

This function is designed to work with tensors where each row represents a
pair of indices. It removes rows containing duplicate entries in the
specified column.

#### Arguments

- index_pairs (torch.Tensor): A 2D tensor of shape [n, 2], where n is the
    number of index pairs.
- index_to_remove (int): The column index (0 or 1) from which to remove
    duplicate entries.

#### Returns

- `-` *torch.Tensor* - A 2D tensor of index pairs with duplicates removed from
    the specified column.

#### Examples

index_pairs = torch.tensor([[1, 2], [3, 4], [1, 2]])
index_to_remove = 0
# Output will be [[1, 2], [3, 4]] assuming column 0 is chosen for removing
    duplicates.

#### Signature

```python
def remove_duplicates(
    index_pairs: torch.Tensor, index_to_remove: int
) -> torch.Tensor: ...
```



## single_axis_sweep_and_prune

[Show source in particle_pairs.py:118](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_pairs.py#L118)

Sweep and prune algorithm for collision detection along a single axis.
This function identifies pairs of particles that are close enough to
potentially collide.

#### Arguments

- `position_axis` *torch.Tensor* - The position of particles along a single
    axis.
- `radius` *torch.Tensor* - The radius of particles.

#### Returns

- `Tuple[torch.Tensor,` *torch.Tensor]* - Two tensors containing the indices
of potentially colliding particles.

#### Signature

```python
def single_axis_sweep_and_prune(
    position_axis: torch.Tensor, radius: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]: ...
```



## validate_pair_distance

[Show source in particle_pairs.py:79](https://github.com/uncscode/particula/blob/main/particula/lagrangian/particle_pairs.py#L79)

Validates if the Euclidean distances between pairs of points are smaller
than the sum of their radii.

#### Arguments

- `collision_indices_pairs` *torch.Tensor* - A tensor containing pairs of
    indices of potentially colliding particles.
- `position` *torch.Tensor* - A 2D tensor of particle positions, where each
    column represents a particle, and each row represents an axis.
- `radius` *torch.Tensor* - A 1D tensor representing the radius of each
    particle.

#### Returns

- `torch.Tensor` - A tensor containing the indices of the pairs of
    particles that are actually colliding.

#### Signature

```python
def validate_pair_distance(
    collision_indices_pairs: torch.Tensor, position: torch.Tensor, radius: torch.Tensor
) -> torch.Tensor: ...
```
