# Collisions

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Lagrangian](./index.md#lagrangian) / Collisions

> Auto-generated documentation for [particula.lagrangian.collisions](https://github.com/uncscode/particula/blob/main/particula/lagrangian/collisions.py) module.

## coalescence

[Show source in collisions.py:69](https://github.com/uncscode/particula/blob/main/particula/lagrangian/collisions.py#L69)

Update mass and velocity of particles based on collision pairs, conserving
mass and momentum.

This function processes collision pairs, sorts them to avoid duplicate
handling, and then updates the mass and velocity of colliding particles
according to the conservation of mass and momentum.

#### Arguments

- `position` *torch.Tensor* - A 2D tensor of shape [n_dimensions, n_particles]
    representing the positions of particles.
- `velocity` *torch.Tensor* - A 2D tensor of shape [n_dimensions, n_particles]
    representing the velocities of particles.
- `mass` *torch.Tensor* - A 1D tensor containing the mass of each particle.
- `radius` *torch.Tensor* - A 1D tensor containing the radius of each particle.
- `collision_indices_pairs` *torch.Tensor* - A 2D tensor containing pairs of
    indices representing colliding particles.
- `remove_duplicates_func` *function* - A function to remove duplicate entries
    from a tensor of index pairs.

#### Returns

- `-` *torch.Tensor* - A 2D tensor of shape [n_dimensions, n_particles]
    representing the updated velocities of particles.

#### Notes

- This function modifies the `velocity` and `mass` tensors in-place.
- It assumes that the mass and momentum are transferred from the right
    particle to the left in each collision pair.
- The subtraction approach for the right-side particles ensures no mass is
    lost in multi-particle collisions (e.g., A<-B and B<-D).

#### Signature

```python
def coalescence(
    position: torch.Tensor,
    velocity: torch.Tensor,
    mass: torch.Tensor,
    radius: torch.Tensor,
    collision_indices_pairs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
```



## elastic_collision

[Show source in collisions.py:136](https://github.com/uncscode/particula/blob/main/particula/lagrangian/collisions.py#L136)

Update velocities of particles based on elastic collision pairs using
matrix operations, conserving kinetic energy and momentum.

#### Arguments

- `velocity` *torch.Tensor* - A 2D tensor of shape [n_dimensions, n_particles]
    representing the velocities of particles.
- `mass` *torch.Tensor* - A 1D tensor containing the mass of each particle.
- `collision_indices_pairs` *torch.Tensor* - A 2D tensor containing pairs of
    indices representing colliding particles.
- `remove_duplicates_func` *function* - A function to remove duplicate entries
    from a tensor of index pairs.

#### Returns

- `torch.Tensor` - A 2D tensor of shape [n_dimensions, n_particles]
    representing the updated velocities of particles.

#### Notes

- This function modifies the `velocity` tensor in-place.
- Mass remains unchanged in elastic collisions.

#### Examples

- `2d` - https://www.wolframalpha.com/input?i=elastic+collision&assumption=%7B%22F%22%2C+%22ElasticCollision%22%2C+%22m2%22%7D+-%3E%221+kg%22&assumption=%7B%22F%22%2C+%22ElasticCollision%22%2C+%22m1%22%7D+-%3E%221+kg%22&assumption=%22FSelect%22+-%3E+%7B%7B%22ElasticCollision2D%22%7D%7D&assumption=%7B%22F%22%2C+%22ElasticCollision%22%2C+%22v1i%22%7D+-%3E%221+m%2Fs%22&assumption=%7B%22F%22%2C+%22ElasticCollision%22%2C+%22v2i%22%7D+-%3E%22-0.5+m%2Fs%22
- `3d` *fortran* - https://www.plasmaphysics.org.uk/programs/coll3d_for.htm
https://www.plasmaphysics.org.uk/collision3d.htm

I think the approach is take a pair, and rotate the coordinate system so
that the collision is in the x-y plane. Then, the z component of the
velocity is a 1d problem, and the x-y component is a 2d problem. Then,
rotate back to the original coordinate system.

#### Signature

```python
def elastic_collision(
    velocity: torch.Tensor, mass: torch.Tensor, collision_indices_pairs: torch.Tensor
) -> torch.Tensor: ...
```



## find_collisions

[Show source in collisions.py:9](https://github.com/uncscode/particula/blob/main/particula/lagrangian/collisions.py#L9)

Find the collision pairs from a distance matrix, given the mass and
indices of particles.

This function identifies pairs of particles that are within a certain
distance threshold (<0), indicating a collision.
It filters out pairs involving particles with zero mass.

#### Arguments

- `distance_matrix` *torch.Tensor* - A 2D tensor containing the pairwise
    distances between particles.
- `indices` *torch.Tensor* - A 1D tensor containing the indices of the
    particles.
- `mass` *torch.Tensor* - A 1D tensor containing the mass of each particle.
- `k` *int, optional* - The number of closest neighbors to consider for each
    particle. Defaults to 1.

#### Returns

- `torch.Tensor` - A 2D tensor of shape [n_collisions, 2] containing the
indices of colliding pairs of particles.

#### Notes

- The function assumes that the diagonal elements of the distance matrix
(distances of particles to themselves) are less than zero.
- Particles with zero mass are excluded from the collision pairs.

#### Signature

```python
def find_collisions(
    distance_matrix: torch.Tensor, indices: torch.Tensor, mass: torch.Tensor, k: int = 1
) -> torch.Tensor: ...
```
