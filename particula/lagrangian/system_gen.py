"""Exploring the pytorch package for lagrangian particle systems."""
# %% imports
import torch
import numpy as np
import matplotlib.pyplot as plt

t_gen = torch.Generator()
t_type = torch.float32

TOTAL_NUMBER_OF_PARTICLES = 500
TIME_STEP = 0.01
SIMULATION_TIME = 10
MASS = 3
DOMAIN_RADIUS = 100

position = torch.rand(3, TOTAL_NUMBER_OF_PARTICLES, dtype=t_type) * DOMAIN_RADIUS
velocity = torch.rand(3, TOTAL_NUMBER_OF_PARTICLES, dtype=t_type) * 10
force = torch.zeros(3, TOTAL_NUMBER_OF_PARTICLES, dtype=t_type)

mass = torch.ones(TOTAL_NUMBER_OF_PARTICLES, dtype=t_type) * MASS
charge = torch.ones(TOTAL_NUMBER_OF_PARTICLES, dtype=t_type) * 1
density = torch.ones(TOTAL_NUMBER_OF_PARTICLES, dtype=t_type) * 1
indices = torch.arange(TOTAL_NUMBER_OF_PARTICLES, dtype=t_type)  # could be int
ids_int = torch.randint(low=0,
                        high=100000,
                        size=(TOTAL_NUMBER_OF_PARTICLES,),
                        generator=t_gen,
                        dtype=torch.int64)
total_iterations = int(SIMULATION_TIME / TIME_STEP)
total_mass = torch.zeros(total_iterations, dtype=t_type)

position.requires_grad = True
velocity.requires_grad = True
mass.requires_grad = False

gravity = torch.tensor([0, -9.81, 0]).repeat(TOTAL_NUMBER_OF_PARTICLES, 1).transpose(0, 1)

# save initial position
position_initial= position.detach().numpy()
velocity_initial= velocity.detach().numpy()
mass_initial = mass.detach().numpy()

# radius of the sphere particles
def torch_radius(mass, density):
    """Calculate radius of sphere particle from mass and density."""
    return torch.pow(3 * mass / (4 * np.pi * density), 1 / 3)


def remove_duplicates(index_pairs, index_to_remove):
    """Removes duplicates from index_pairs, from the index_to_remove column."""
    # Sort index_pairs by the index_to_remove column
    sorted_index_pairs, indices_sorted = torch.sort(index_pairs[:, index_to_remove], dim=0)
    # Find duplicates in the index_to_remove column
    diff_index = torch.diff(sorted_index_pairs, prepend=torch.ones(1)*-1) > 0
    # Remove duplicates indices
    clean_index = indices_sorted[diff_index]
    return index_pairs[clean_index, :]

for i in range(total_iterations):
    # calculate collisions of particles with each other, Find nearest neighbours

    # update position from velocity using leapfrog
    position = position \
        + TIME_STEP * velocity \
        + 0.5 * TIME_STEP * TIME_STEP * force / mass
    # update velocity from force using leapfrog
    velocity = velocity \
        + 0.5 * TIME_STEP * force / mass

    # calculate force for gravity
    force = mass * gravity

    # Collisions, Calculate pairwise distance
    # Expand position tensor and compute differences
    diff = position.unsqueeze(2) - position.unsqueeze(1)
    # Sum square differences and take square root to get Euclidean distances
    distance_matrix = torch.sqrt(torch.sum(diff**2, dim=[0]))

    # Number of neighbors
    k = 1

    # Top k closest particle (including self)
    closest_neighbors = torch.topk(
        distance_matrix, k=k + 1, largest=False, sorted=True)
    collision_distance = 2 * torch_radius(mass, density)
    # Find collisions
    collisions = closest_neighbors.values < collision_distance.unsqueeze(1)
    # remove zero mass particles
    collisions = collisions * (mass.unsqueeze(1) > 0)
    # unique indices in collisions
    iteration_index = 1
    expanded_indices = indices.unsqueeze(1).repeat(1, k + 1)
    expanded_closest_neighbors = closest_neighbors.indices

    collision_indices_pairs = torch.cat([
        expanded_closest_neighbors.unsqueeze(2),
        expanded_indices.unsqueeze(2)],
        dim=2)
    valid_collision_indices_pairs = collision_indices_pairs[
        collisions[:, iteration_index], iteration_index, :].int()

    # from the possible collisions, move the lowest index to the left
    sorted_pairs, _ = torch.sort(valid_collision_indices_pairs, dim=1)
    # remove duplicates along the left column
    unique_left_indices = remove_duplicates(sorted_pairs, 0)
    # remove duplicates along the right column
    unique_indices = remove_duplicates(unique_left_indices, 1)
    # a triple collision will not be moved in a single iteration
    # only one collision will be moved per iteration

    mass[unique_indices[:, 0]] += mass[unique_indices[:, 1]]  # mass gain
    mass[unique_indices[:, 1]] -= mass[unique_indices[:, 1]]  # mass lost
    # better than zeroing incase of a triple collision, the mass won't be lost
    total_mass[i] = torch.sum(mass)

print('done')
# %% plot results
position_final= position.detach().numpy()
velocity_final= velocity.detach().numpy()
mass = mass.detach().numpy()
total_mass = total_mass.detach().numpy()
# calculate speed
speed_initial = np.linalg.norm(velocity_initial, axis=0)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Define color map range
cmap = plt.cm.viridis  # You can choose any other colormap
vmin = mass.min()
vmax = mass.max()

# Plot initial positions
scatter1 = ax.scatter(
    position_initial[0],
    position_initial[1],
    position_initial[2],
    c=mass_initial,
    cmap=cmap,
    vmin=vmin,
    vmax=vmax)

# Plot final positions
scatter2 = ax.scatter(
    position_final[0],
    position_final[1],
    position_final[2],
    c=mass,
    cmap=cmap,
    vmin=vmin,
    vmax=vmax)

# Create color bar
cbar = plt.colorbar(scatter2, ax=ax)
cbar.set_label('Mass')
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.hist(mass_initial, bins=10) 
ax.hist(mass, bins=10, alpha=0.5)
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(total_mass)
plt.show()
print('done')