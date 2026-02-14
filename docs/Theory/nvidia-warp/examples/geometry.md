# Particle Interactions and Wall Loss with NVIDIA Warp

## Overview

This guide demonstrates how to implement particle-wall interactions, chamber geometry, and wall loss calculations using NVIDIA Warp. These processes are critical for accurate aerosol chamber simulations.

## Particle-Wall Collisions

### Spherical Chamber

```python
import warp as wp

@wp.func
def distance_to_sphere_wall(
    position: wp.vec3,
    sphere_center: wp.vec3,
    sphere_radius: float,
) -> float:
    """Distance from particle to spherical chamber wall."""
    r = wp.length(position - sphere_center)
    return sphere_radius - r

@wp.kernel
def apply_sphere_boundary(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    sphere_center: wp.vec3,
    sphere_radius: float,
    restitution: float,
):
    """Reflect particles off spherical chamber walls."""
    tid = wp.tid()
    
    pos = positions[tid]
    vel = velocities[tid]
    
    # Vector from center to particle
    r_vec = pos - sphere_center
    r = wp.length(r_vec)
    
    # Check if outside sphere
    if r > sphere_radius:
        # Normal pointing inward
        normal = -r_vec / r
        
        # Push particle back inside
        pos = sphere_center + r_vec * (sphere_radius / r)
        
        # Reflect velocity
        vel_normal = wp.dot(vel, normal)
        if vel_normal < 0.0:  # Moving toward wall
            vel = vel - (1.0 + restitution) * vel_normal * normal
    
    positions[tid] = pos
    velocities[tid] = vel
```

### Rectangular Chamber

```python
@wp.kernel
def apply_box_boundary(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    box_min: wp.vec3,
    box_max: wp.vec3,
    restitution: float,
):
    """Reflect particles off rectangular chamber walls."""
    tid = wp.tid()
    
    pos = positions[tid]
    vel = velocities[tid]
    
    # Check each axis
    # X-axis walls
    if pos[0] < box_min[0]:
        pos = wp.vec3(box_min[0], pos[1], pos[2])
        if vel[0] < 0.0:
            vel = wp.vec3(-restitution * vel[0], vel[1], vel[2])
    elif pos[0] > box_max[0]:
        pos = wp.vec3(box_max[0], pos[1], pos[2])
        if vel[0] > 0.0:
            vel = wp.vec3(-restitution * vel[0], vel[1], vel[2])
    
    # Y-axis walls
    if pos[1] < box_min[1]:
        pos = wp.vec3(pos[0], box_min[1], pos[2])
        if vel[1] < 0.0:
            vel = wp.vec3(vel[0], -restitution * vel[1], vel[2])
    elif pos[1] > box_max[1]:
        pos = wp.vec3(pos[0], box_max[1], pos[2])
        if vel[1] > 0.0:
            vel = wp.vec3(vel[0], -restitution * vel[1], vel[2])
    
    # Z-axis walls
    if pos[2] < box_min[2]:
        pos = wp.vec3(pos[0], pos[1], box_min[2])
        if vel[2] < 0.0:
            vel = wp.vec3(vel[0], vel[1], -restitution * vel[2])
    elif pos[2] > box_max[2]:
        pos = wp.vec3(pos[0], pos[1], box_max[2])
        if vel[2] > 0.0:
            vel = wp.vec3(vel[0], vel[1], -restitution * vel[2])
    
    positions[tid] = pos
    velocities[tid] = vel
```

### Cylindrical Chamber

```python
@wp.kernel
def apply_cylinder_boundary(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    cylinder_center: wp.vec3,
    cylinder_radius: float,
    cylinder_height: float,
    restitution: float,
):
    """Reflect particles off cylindrical chamber walls.
    
    Cylinder axis is along z-direction.
    """
    tid = wp.tid()
    
    pos = positions[tid]
    vel = velocities[tid]
    
    # Radial distance from axis (in x-y plane)
    dx = pos[0] - cylinder_center[0]
    dy = pos[1] - cylinder_center[1]
    r = wp.sqrt(dx*dx + dy*dy)
    
    # Radial wall (curved surface)
    if r > cylinder_radius:
        # Radial normal
        normal_x = dx / r
        normal_y = dy / r
        
        # Push inside
        pos = wp.vec3(
            cylinder_center[0] + normal_x * cylinder_radius,
            cylinder_center[1] + normal_y * cylinder_radius,
            pos[2]
        )
        
        # Reflect radial velocity component
        vel_radial = vel[0] * normal_x + vel[1] * normal_y
        if vel_radial > 0.0:
            vel = wp.vec3(
                vel[0] - (1.0 + restitution) * vel_radial * normal_x,
                vel[1] - (1.0 + restitution) * vel_radial * normal_y,
                vel[2]
            )
    
    # Top and bottom walls
    z_min = cylinder_center[2] - cylinder_height / 2.0
    z_max = cylinder_center[2] + cylinder_height / 2.0
    
    if pos[2] < z_min:
        pos = wp.vec3(pos[0], pos[1], z_min)
        if vel[2] < 0.0:
            vel = wp.vec3(vel[0], vel[1], -restitution * vel[2])
    elif pos[2] > z_max:
        pos = wp.vec3(pos[0], pos[1], z_max)
        if vel[2] > 0.0:
            vel = wp.vec3(vel[0], vel[1], -restitution * vel[2])
    
    positions[tid] = pos
    velocities[tid] = vel
```

## Wall Loss Rate Calculations

Wall loss is the deposition of particles on chamber surfaces due to diffusion, settling, and electrostatic effects.

### Diffusion-Limited Wall Loss

```python
# Physical constants — in practice, pass from particula.util.constants
# as kernel parameters. Module-level constants shown here for brevity.
PI = 3.141592653589793  # matches math.pi exactly

@wp.func
def diffusion_wall_loss_rate_sphere(
    diameter: float,
    diffusion_coeff: float,
    chamber_radius: float,
) -> float:
    """Diffusion-limited wall loss rate for spherical chamber.
    
    Based on Crump & Seinfeld (1981).
    Returns loss rate coefficient (1/s).
    """
    # First-order loss rate: beta = D * pi^2 / R^2
    return diffusion_coeff * PI * PI / (chamber_radius * chamber_radius)

@wp.func
def diffusion_wall_loss_rate_box(
    diameter: float,
    diffusion_coeff: float,
    length: float,
    width: float,
    height: float,
) -> float:
    """Diffusion-limited wall loss rate for rectangular chamber."""
    # Sum of contributions from each dimension
    L2 = length * length
    W2 = width * width
    H2 = height * height
    
    return diffusion_coeff * PI * PI * (1.0/L2 + 1.0/W2 + 1.0/H2)

@wp.kernel
def compute_diffusion_wall_loss(
    diameters: wp.array(dtype=float),
    diffusion_coeffs: wp.array(dtype=float),
    wall_loss_rates: wp.array(dtype=float),
    chamber_radius: float,
):
    """Compute diffusion wall loss rate for each particle."""
    tid = wp.tid()
    
    D = diffusion_coeffs[tid]
    wall_loss_rates[tid] = diffusion_wall_loss_rate_sphere(
        diameters[tid], D, chamber_radius
    )
```

### Gravitational Settling Wall Loss

```python
@wp.func
def settling_velocity(
    diameter: float,
    particle_density: float,
    air_density: float,
    viscosity: float,
    slip_correction: float,
) -> float:
    """Terminal settling velocity (Stokes regime with slip correction)."""
    g = 9.81  # m/s²
    return (particle_density - air_density) * g * diameter * diameter * \
           slip_correction / (18.0 * viscosity)

@wp.kernel
def compute_settling_wall_loss(
    diameters: wp.array(dtype=float),
    densities: wp.array(dtype=float),
    slip_corrections: wp.array(dtype=float),
    wall_loss_rates: wp.array(dtype=float),
    chamber_height: float,
    air_density: float,
    viscosity: float,
):
    """Compute gravitational settling wall loss rate.
    
    Assumes particles settle to bottom of chamber.
    """
    tid = wp.tid()
    
    v_s = settling_velocity(
        diameters[tid], densities[tid], air_density,
        viscosity, slip_corrections[tid]
    )
    
    # Loss rate = v_s / H (for well-mixed chamber)
    wall_loss_rates[tid] = v_s / chamber_height
```

### Combined Wall Loss

```python
@wp.kernel
def compute_total_wall_loss_rate(
    diameters: wp.array(dtype=float),
    densities: wp.array(dtype=float),
    diffusion_coeffs: wp.array(dtype=float),
    slip_corrections: wp.array(dtype=float),
    total_loss_rates: wp.array(dtype=float),
    chamber_radius: float,
    chamber_height: float,
    air_density: float,
    viscosity: float,
    eddy_diffusivity: float,
):
    """Compute total wall loss rate including all mechanisms.
    
    Combines diffusion, settling, and turbulent deposition.
    """
    tid = wp.tid()
    
    d = diameters[tid]
    D = diffusion_coeffs[tid]
    Cc = slip_corrections[tid]
    rho_p = densities[tid]
    
    # Diffusion loss
    beta_diff = diffusion_wall_loss_rate_sphere(d, D, chamber_radius)
    
    # Settling loss (only contributes to floor)
    v_s = settling_velocity(d, rho_p, air_density, viscosity, Cc)
    beta_settle = v_s / chamber_height
    
    # Turbulent deposition enhancement
    # Approximation: use eddy diffusivity to enhance diffusion
    D_eff = D + eddy_diffusivity
    beta_turb = diffusion_wall_loss_rate_sphere(d, D_eff, chamber_radius)
    
    # Total (not simply additive, use maximum as approximation)
    total_loss_rates[tid] = beta_turb + beta_settle

@wp.kernel
def apply_wall_loss(
    concentrations: wp.array(dtype=float),
    loss_rates: wp.array(dtype=float),
    dt: float,
):
    """Apply wall loss to particle concentrations.
    
    Uses first-order decay: n(t+dt) = n(t) * exp(-beta * dt)
    """
    tid = wp.tid()
    
    beta = loss_rates[tid]
    concentrations[tid] = concentrations[tid] * wp.exp(-beta * dt)
```

## Electrostatic Wall Loss

For charged particles in electric fields:

```python
@wp.func
def electrical_mobility(
    diameter: float,
    n_charges: int,
    viscosity: float,
    slip_correction: float,
) -> float:
    """Electrical mobility of charged particle.
    
    Z = n * e * Cc / (3 * pi * mu * d)
    """
    e = 1.602e-19  # Elementary charge (C)
    return float(n_charges) * e * slip_correction / \
           (3.0 * PI * viscosity * diameter)

@wp.func
def image_charge_velocity(
    distance_to_wall: float,
    diameter: float,
    n_charges: int,
    viscosity: float,
    slip_correction: float,
    dielectric_constant: float,
) -> float:
    """Velocity toward wall due to image charge attraction.
    
    Important for charged particles near conducting walls.
    """
    e = 1.602e-19
    epsilon_0 = 8.854e-12  # Vacuum permittivity
    
    # Image force: F = n²e² / (16 * pi * eps_0 * eps_r * d²)
    # where d is distance to wall
    n = float(n_charges)
    F_image = n * n * e * e / \
              (16.0 * PI * epsilon_0 * dielectric_constant * \
               distance_to_wall * distance_to_wall)
    
    # Velocity = F * mobility / charge
    Z = electrical_mobility(diameter, n_charges, viscosity, slip_correction)
    return F_image * Z / (n * e) if n > 0 else 0.0

@wp.kernel
def compute_electrostatic_wall_loss(
    positions: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    charges: wp.array(dtype=int),
    slip_corrections: wp.array(dtype=float),
    wall_loss_rates: wp.array(dtype=float),
    chamber_center: wp.vec3,
    chamber_radius: float,
    viscosity: float,
    wall_potential: float,
    dielectric_constant: float,
):
    """Compute electrostatic enhancement to wall loss."""
    tid = wp.tid()
    
    pos = positions[tid]
    d = diameters[tid]
    n_charges = charges[tid]
    Cc = slip_corrections[tid]
    
    if n_charges == 0:
        wall_loss_rates[tid] = 0.0
        return
    
    # Distance to wall
    r = wp.length(pos - chamber_center)
    dist_to_wall = chamber_radius - r
    
    # Clamp minimum distance to particle radius
    if dist_to_wall < d / 2.0:
        dist_to_wall = d / 2.0
    
    # Image charge velocity
    v_image = image_charge_velocity(
        dist_to_wall, d, n_charges, viscosity, Cc, dielectric_constant
    )
    
    # Electric field drift (if wall has potential)
    Z = electrical_mobility(d, n_charges, viscosity, Cc)
    E_field = wall_potential / chamber_radius  # Simplified
    v_drift = Z * E_field
    
    # Total electrostatic deposition velocity
    v_elec = v_image + v_drift
    
    # Convert to loss rate (approximate)
    wall_loss_rates[tid] = v_elec / dist_to_wall
```

## Particle-Particle Interactions

### Hard Sphere Collisions

```python
@wp.func
def hard_sphere_collision(
    pos_i: wp.vec3, vel_i: wp.vec3, mass_i: float, radius_i: float,
    pos_j: wp.vec3, vel_j: wp.vec3, mass_j: float, radius_j: float,
    restitution: float,
) -> wp.vec3:
    """Compute velocity change from hard sphere collision."""
    r_vec = pos_j - pos_i
    dist = wp.length(r_vec)
    collision_dist = radius_i + radius_j
    
    if dist >= collision_dist or dist < 1e-12:
        return wp.vec3(0.0, 0.0, 0.0)
    
    # Collision normal
    normal = r_vec / dist
    
    # Relative velocity
    vel_rel = vel_i - vel_j
    vel_normal = wp.dot(vel_rel, normal)
    
    # Only collide if approaching
    if vel_normal <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    
    # Impulse magnitude (elastic collision with restitution)
    j = -(1.0 + restitution) * vel_normal / (1.0/mass_i + 1.0/mass_j)
    
    # Velocity change for particle i
    return -normal * (j / mass_i)

@wp.kernel
def detect_particle_collisions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    masses: wp.array(dtype=float),
    velocity_changes: wp.array(dtype=wp.vec3),
    grid: wp.uint64,
    search_radius: float,
    restitution: float,
):
    """Detect and resolve particle-particle collisions."""
    tid = wp.tid()
    
    pos_i = positions[tid]
    vel_i = velocities[tid]
    r_i = radii[tid]
    m_i = masses[tid]
    
    delta_v = wp.vec3(0.0, 0.0, 0.0)
    
    # Query neighbors
    query = wp.hash_grid_query(grid, pos_i, search_radius)
    j = wp.hash_grid_query_next(query, grid)
    
    while j >= 0:
        if j != tid:
            delta_v += hard_sphere_collision(
                pos_i, vel_i, m_i, r_i,
                positions[j], velocities[j], masses[j], radii[j],
                restitution
            )
        j = wp.hash_grid_query_next(query, grid)
    
    velocity_changes[tid] = delta_v
```

## Complete Chamber Simulation

```python
def simulate_chamber(
    n_particles: int = 10000,
    n_steps: int = 1000,
    dt: float = 0.01,
    chamber_type: str = "sphere",
):
    """Complete chamber simulation with wall loss."""
    wp.init()
    
    # Chamber geometry
    if chamber_type == "sphere":
        chamber_radius = 0.5  # 0.5 m radius
        chamber_volume = (4.0/3.0) * 3.14159 * chamber_radius**3
    else:  # box
        box_size = 1.0  # 1 m cube
        chamber_volume = box_size**3
    
    # Environmental conditions
    temperature = 298.15  # K
    pressure = 101325.0   # Pa
    viscosity = 1.81e-5   # Pa·s
    air_density = 1.2     # kg/m³
    mean_free_path = 6.8e-8  # m
    
    # Initialize particles
    positions = wp.array(
        np.random.uniform(-0.4, 0.4, (n_particles, 3)),
        dtype=wp.vec3, device="cuda"
    )
    velocities = wp.zeros(n_particles, dtype=wp.vec3, device="cuda")
    diameters = wp.array(
        np.random.lognormal(np.log(100e-9), 0.5, n_particles),
        dtype=float, device="cuda"
    )
    densities = wp.array(
        np.ones(n_particles) * 1500.0,  # kg/m³
        dtype=float, device="cuda"
    )
    concentrations = wp.array(
        np.ones(n_particles) * 1e6,
        dtype=float, device="cuda"
    )
    
    # Derived properties
    diffusion_coeffs = wp.zeros(n_particles, dtype=float, device="cuda")
    slip_corrections = wp.zeros(n_particles, dtype=float, device="cuda")
    wall_loss_rates = wp.zeros(n_particles, dtype=float, device="cuda")
    
    # Pre-compute slip corrections and diffusion coefficients
    # (would use kernels from previous examples)
    
    chamber_center = wp.vec3(0.0, 0.0, 0.0)
    
    for step in range(n_steps):
        # 1. Apply Brownian motion (from previous examples)
        
        # 2. Apply gravitational settling
        wp.launch(
            compute_settling_wall_loss,
            dim=n_particles,
            inputs=[
                diameters, densities, slip_corrections,
                wall_loss_rates, chamber_radius,
                air_density, viscosity
            ]
        )
        
        # 3. Apply wall loss to concentrations
        wp.launch(
            apply_wall_loss,
            dim=n_particles,
            inputs=[concentrations, wall_loss_rates, dt]
        )
        
        # 4. Apply boundary conditions
        if chamber_type == "sphere":
            wp.launch(
                apply_sphere_boundary,
                dim=n_particles,
                inputs=[
                    positions, velocities,
                    chamber_center, chamber_radius, 0.0
                ]
            )
        
        wp.synchronize()
        
        if step % 100 == 0:
            total_conc = np.sum(concentrations.numpy())
            print(f"Step {step}: total concentration = {total_conc:.2e} #/m³")
    
    return {
        "positions": positions.numpy(),
        "concentrations": concentrations.numpy(),
        "diameters": diameters.numpy(),
    }
```

## Performance Tips

### Spatial Hashing for Wall Distance

```python
# Pre-compute wall distances using spatial grid
wall_grid = wp.HashGrid(dim_x=32, dim_y=32, dim_z=32)

@wp.kernel
def compute_wall_distances_fast(
    positions: wp.array(dtype=wp.vec3),
    wall_distances: wp.array(dtype=float),
    chamber_center: wp.vec3,
    chamber_radius: float,
):
    """Vectorized wall distance computation."""
    tid = wp.tid()
    pos = positions[tid]
    r = wp.length(pos - chamber_center)
    wall_distances[tid] = chamber_radius - r
```

### Batch Processing for Large Populations

```python
# Process particles in batches to manage GPU memory
batch_size = 100000

for batch_start in range(0, n_particles, batch_size):
    batch_end = min(batch_start + batch_size, n_particles)
    batch_n = batch_end - batch_start
    
    wp.launch(
        wall_loss_kernel,
        dim=batch_n,
        inputs=[
            positions[batch_start:batch_end],
            # ... other arrays
        ]
    )
```

## See Also

- [Kernels and Function Inlining](../kernels.md) - Kernel optimization
- [Data Structures](../datastructures.md) - Particle data types  
- [Aerosol Dynamics](./fluids.md) - Coagulation, condensation
