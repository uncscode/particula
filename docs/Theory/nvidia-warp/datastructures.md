# Data Structures for Aerosol Simulation

## Overview

NVIDIA Warp provides a rich set of data structures optimized for parallel computation on both CPU and GPU. For particula, these structures efficiently represent particle populations, aerosol properties, and simulation state.

## Arrays

### Array Types

Warp arrays are the primary container for particle data:

```python
import warp as wp
import numpy as np

# 1D array for particle diameters
diameters = wp.array([1e-6, 2e-6, 5e-6], dtype=float)

# From NumPy (common for initialization)
concentrations = np.random.uniform(1e6, 1e8, size=1000)  # #/m³
conc_array = wp.array(concentrations, dtype=float)

# Multi-dimensional arrays for coagulation kernels
kernel_matrix = wp.zeros((100, 100), dtype=float)
```

### Array Properties

```python
# Shape
print(diameters.shape)  # (3,)

# Data type
print(diameters.dtype)  # float32

# Device (CPU or GPU)
print(diameters.device)  # "cpu" or "cuda"

# Size (number of particles)
print(diameters.size)  # 3
```

### Array Operations

```python
# Create arrays on GPU for large particle populations
n_particles = 100000
gpu_positions = wp.zeros(n_particles, dtype=wp.vec3, device="cuda")
gpu_diameters = wp.zeros(n_particles, dtype=float, device="cuda")

# Copy between CPU and GPU
cpu_data = wp.array(np.ones(n_particles) * 1e-6, dtype=float)
gpu_diameters.assign(cpu_data)

# Convert back to NumPy for analysis
result = gpu_diameters.numpy()
```

## Vector Types

### Particle Position and Velocity

Warp provides fixed-size vector types ideal for 3D particle tracking:

```python
# 3D position vector (meters)
position = wp.vec3(0.0, 0.0, 0.0)

# 3D velocity vector (m/s)
velocity = wp.vec3(1e-3, 0.0, -1e-4)

# 2D for simplified simulations
position_2d = wp.vec2(0.0, 0.0)
```

### Vector Operations in Kernels

```python
@wp.kernel
def update_particle_positions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
):
    tid = wp.tid()
    
    # Vector addition
    positions[tid] = positions[tid] + velocities[tid] * dt

@wp.kernel
def compute_settling_velocities(
    diameters: wp.array(dtype=float),
    densities: wp.array(dtype=float),
    velocities: wp.array(dtype=wp.vec3),
    air_density: float,
    viscosity: float,
):
    tid = wp.tid()
    
    d = diameters[tid]
    rho_p = densities[tid]
    
    # Stokes settling velocity (downward)
    g = 9.81  # m/s²
    v_settle = (rho_p - air_density) * g * d * d / (18.0 * viscosity)
    
    # Set velocity vector (settling in -z direction)
    velocities[tid] = wp.vec3(0.0, 0.0, -v_settle)
```

### Distance Calculations

```python
@wp.func
def particle_separation(pos_i: wp.vec3, pos_j: wp.vec3) -> float:
    """Calculate distance between two particles."""
    diff = pos_j - pos_i
    return wp.length(diff)

@wp.func
def unit_direction(pos_i: wp.vec3, pos_j: wp.vec3) -> wp.vec3:
    """Unit vector from particle i to particle j."""
    diff = pos_j - pos_i
    dist = wp.length(diff)
    if dist > 1e-12:
        return diff / dist
    return wp.vec3(0.0, 0.0, 0.0)
```

## Matrix Types

### Coagulation Kernel Matrices

```python
@wp.kernel
def build_coagulation_matrix(
    diameters: wp.array(dtype=float),
    kernel_matrix: wp.array2d(dtype=float),
    temperature: float,
    viscosity: float,
):
    """Build pairwise coagulation kernel matrix K[i,j]."""
    i, j = wp.tid()
    
    d_i = diameters[i]
    d_j = diameters[j]
    
    # Brownian coagulation kernel
    k_B = 1.380649e-23
    K_ij = (2.0 * k_B * temperature / (3.0 * viscosity)) * \
           (d_i + d_j) * (1.0/d_i + 1.0/d_j)
    
    kernel_matrix[i, j] = K_ij
```

### Transformation Matrices

For rotating reference frames or coordinate transformations:

```python
@wp.func
def rotation_matrix_z(angle: float) -> wp.mat33:
    """Rotation matrix about z-axis."""
    c = wp.cos(angle)
    s = wp.sin(angle)
    return wp.mat33(c, -s, 0.0,
                    s,  c, 0.0,
                    0.0, 0.0, 1.0)

@wp.kernel
def rotate_velocities(
    velocities: wp.array(dtype=wp.vec3),
    angle: float,
):
    """Rotate all particle velocities."""
    tid = wp.tid()
    R = rotation_matrix_z(angle)
    velocities[tid] = R * velocities[tid]
```

## Custom Structs

### Particle Structure

Define custom data structures for aerosol particles:

```python
@wp.struct
class Particle:
    """Single aerosol particle properties."""
    position: wp.vec3      # Position (m)
    velocity: wp.vec3      # Velocity (m/s)
    diameter: float        # Diameter (m)
    mass: float           # Mass (kg)
    density: float        # Material density (kg/m³)
    charge: int           # Number of elementary charges

@wp.struct
class ParticleSpecies:
    """Properties of a particle species/component."""
    molar_mass: float      # Molar mass (kg/mol)
    density: float         # Bulk density (kg/m³)
    surface_tension: float # Surface tension (N/m)
    vapor_pressure: float  # Saturation vapor pressure (Pa)
    hygroscopicity: float  # Kappa hygroscopicity parameter

@wp.kernel
def update_particles(
    particles: wp.array(dtype=Particle),
    dt: float,
    gravity: wp.vec3,
):
    """Update particle positions and velocities."""
    tid = wp.tid()
    p = particles[tid]
    
    # Apply gravity
    acceleration = gravity
    p.velocity = p.velocity + acceleration * dt
    p.position = p.position + p.velocity * dt
    
    particles[tid] = p
```

### Aerosol Distribution Structure

```python
@wp.struct
class SizeDistribution:
    """Log-normal size distribution parameters."""
    geometric_mean: float  # Geometric mean diameter (m)
    geometric_std: float   # Geometric standard deviation
    total_number: float    # Total number concentration (#/m³)

@wp.struct
class AerosolState:
    """Complete aerosol system state."""
    temperature: float     # Temperature (K)
    pressure: float        # Pressure (Pa)
    relative_humidity: float  # Relative humidity (0-1)
    n_particles: int       # Number of particles
```

## Hash Grids for Neighbor Search

### Spatial Hashing for Coagulation

Efficient neighbor searches for particle-particle interactions:

```python
# Create hash grid for spatial queries
grid = wp.HashGrid(dim_x=64, dim_y=64, dim_z=64)

@wp.kernel
def build_particle_grid(
    positions: wp.array(dtype=wp.vec3),
    grid: wp.uint64,
):
    """Build spatial hash grid from particle positions."""
    tid = wp.tid()
    wp.hash_grid_point_id(grid, tid, positions[tid])

@wp.kernel
def find_collision_candidates(
    positions: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    grid: wp.uint64,
    collision_pairs: wp.array(dtype=wp.vec2i),
    n_collisions: wp.array(dtype=int),
    search_radius: float,
):
    """Find particle pairs within collision distance."""
    tid = wp.tid()
    pos_i = positions[tid]
    d_i = diameters[tid]
    
    # Query neighbors within search radius
    query = wp.hash_grid_query(grid, pos_i, search_radius)
    neighbor_idx = wp.hash_grid_query_next(query, grid)
    
    while neighbor_idx >= 0:
        if neighbor_idx > tid:  # Avoid double counting
            pos_j = positions[neighbor_idx]
            d_j = diameters[neighbor_idx]
            
            # Check if particles overlap
            dist = wp.length(pos_j - pos_i)
            collision_dist = 0.5 * (d_i + d_j)
            
            if dist < collision_dist:
                # Record collision pair
                idx = wp.atomic_add(n_collisions, 0, 1)
                collision_pairs[idx] = wp.vec2i(tid, neighbor_idx)
        
        neighbor_idx = wp.hash_grid_query_next(query, grid)
```

## Chamber Geometry

### Mesh for Wall Loss Calculations

```python
# Define chamber walls as triangulated mesh
vertices = wp.array(vertex_data, dtype=wp.vec3)
indices = wp.array(index_data, dtype=int)

chamber_mesh = wp.Mesh(vertices, indices)

@wp.kernel
def compute_wall_distances(
    positions: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    wall_distances: wp.array(dtype=float),
):
    """Compute distance from each particle to nearest wall."""
    tid = wp.tid()
    pos = positions[tid]
    
    # Find closest point on mesh
    face_idx = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    
    if wp.mesh_query_point(mesh, pos, 1e6, sign, face_idx, face_u, face_v):
        # Get closest point on face
        closest = wp.mesh_eval_position(mesh, face_idx, face_u, face_v)
        wall_distances[tid] = wp.length(pos - closest)
    else:
        wall_distances[tid] = 1e6  # No wall found
```

## Performance Tips

### Structure of Arrays vs Array of Structures

```python
# Structure of Arrays (SoA) - Better for GPU
# Each property is a separate array
positions = wp.zeros(n_particles, dtype=wp.vec3)
diameters = wp.zeros(n_particles, dtype=float)
masses = wp.zeros(n_particles, dtype=float)
concentrations = wp.zeros(n_particles, dtype=float)

# Array of Structures (AoS) - More convenient
# Single array of Particle structs
particles = wp.zeros(n_particles, dtype=Particle)
```

**Recommendation**: Use SoA for large particle populations (>10,000) where performance matters. Use AoS for smaller simulations where code clarity is more important.

### Data Transfer Optimization

```python
# BAD: Multiple CPU-GPU transfers per timestep
for step in range(n_steps):
    data = gpu_array.numpy()  # Transfer to CPU
    # ... process on CPU
    gpu_array.assign(data)    # Transfer back to GPU

# GOOD: Keep data on GPU, transfer only when needed
for step in range(n_steps):
    wp.launch(process_kernel, dim=n, inputs=[gpu_array])
    
# Transfer only for output/analysis
if step % output_interval == 0:
    result = gpu_array.numpy()
```

### Pre-allocation

```python
# Pre-allocate all arrays before simulation loop
positions = wp.zeros(n_particles, dtype=wp.vec3)
velocities = wp.zeros(n_particles, dtype=wp.vec3)
diameters = wp.zeros(n_particles, dtype=float)
temp_buffer = wp.zeros(n_particles, dtype=float)  # For intermediate results

# Reuse arrays across timesteps
for step in range(n_steps):
    wp.launch(compute_kernel, dim=n, inputs=[positions, velocities, temp_buffer])
    # temp_buffer is reused each step
```

## See Also

- [Kernels and Function Inlining](./kernels.md) - Writing parallel particle computations
- [Aerosol Dynamics Example](./examples/fluids.md) - Coagulation, condensation
- [Particle Interactions Example](./examples/geometry.md) - Collisions, wall loss
