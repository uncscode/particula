# Aerosol Dynamics with NVIDIA Warp

## Overview

This guide demonstrates how to implement aerosol dynamics simulations using NVIDIA Warp. We cover the core processes: coagulation (particle collisions), condensation/evaporation (mass transfer), and Brownian diffusion.

## Coagulation

Coagulation is the process where particles collide and merge. For aerosols, Brownian motion drives most collisions.

### Brownian Coagulation Kernel

```python
import warp as wp
import numpy as np

# Physical constants — in practice, pass from particula.util.constants
# as kernel parameters. Module-level constants shown here for brevity.
PI = 3.141592653589793  # matches math.pi exactly

@wp.func
def cunningham_correction(diameter: float, mean_free_path: float) -> float:
    """Cunningham slip correction factor for small particles."""
    kn = 2.0 * mean_free_path / diameter
    A1, A2, A3 = 1.257, 0.400, 0.55
    return 1.0 + kn * (A1 + A2 * wp.exp(-A3 / kn))

@wp.func
def diffusion_coefficient(
    diameter: float,
    temperature: float,
    viscosity: float,
    mean_free_path: float,
    k_boltzmann: float,  # from particula.util.constants.BOLTZMANN_CONSTANT
) -> float:
    """Stokes-Einstein diffusion coefficient with slip correction."""
    Cc = cunningham_correction(diameter, mean_free_path)
    return (k_boltzmann * temperature * Cc) / (3.0 * PI * viscosity * diameter)

@wp.func
def brownian_coagulation_kernel(
    d_i: float,
    d_j: float,
    temperature: float,
    viscosity: float,
    mean_free_path: float,
) -> float:
    """Compute Brownian coagulation kernel K(d_i, d_j).
    
    This is the rate coefficient for collision between particles
    of diameters d_i and d_j due to Brownian motion.
    """
    # Diffusion coefficients
    D_i = diffusion_coefficient(d_i, temperature, viscosity, mean_free_path)
    D_j = diffusion_coefficient(d_j, temperature, viscosity, mean_free_path)
    
    # Collision kernel: K = 2*pi*(D_i + D_j)*(d_i + d_j)
    return 2.0 * PI * (D_i + D_j) * (d_i + d_j)
```

### Coagulation Rate Computation

```python
@wp.kernel
def compute_coagulation_rates(
    diameters: wp.array(dtype=float),
    concentrations: wp.array(dtype=float),
    coag_rates: wp.array(dtype=float),
    temperature: float,
    viscosity: float,
    mean_free_path: float,
    n_bins: int,
):
    """Compute coagulation loss rate for each size bin."""
    i = wp.tid()
    d_i = diameters[i]
    n_i = concentrations[i]
    
    rate = 0.0
    
    # Sum over all collision partners
    for j in range(n_bins):
        d_j = diameters[j]
        n_j = concentrations[j]
        
        K_ij = brownian_coagulation_kernel(
            d_i, d_j, temperature, viscosity, mean_free_path
        )
        
        # Loss rate: dn_i/dt = -sum_j(K_ij * n_i * n_j)
        rate -= K_ij * n_i * n_j
    
    coag_rates[i] = rate

@wp.kernel
def build_coagulation_matrix(
    diameters: wp.array(dtype=float),
    kernel_matrix: wp.array2d(dtype=float),
    temperature: float,
    viscosity: float,
    mean_free_path: float,
):
    """Build full coagulation kernel matrix K[i,j]."""
    i, j = wp.tid()
    
    kernel_matrix[i, j] = brownian_coagulation_kernel(
        diameters[i], diameters[j],
        temperature, viscosity, mean_free_path
    )
```

### Particle-Resolved Coagulation

For explicit particle tracking (Monte Carlo approach):

```python
@wp.kernel
def find_coagulation_partners(
    positions: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    grid: wp.uint64,
    partner_indices: wp.array(dtype=int),
    collision_probs: wp.array(dtype=float),
    temperature: float,
    viscosity: float,
    mean_free_path: float,
    dt: float,
    search_radius: float,
):
    """Find collision partners for each particle."""
    tid = wp.tid()
    pos_i = positions[tid]
    d_i = diameters[tid]
    
    best_partner = -1
    best_prob = 0.0
    
    # Query nearby particles
    query = wp.hash_grid_query(grid, pos_i, search_radius)
    neighbor = wp.hash_grid_query_next(query, grid)
    
    while neighbor >= 0:
        if neighbor != tid:
            d_j = diameters[neighbor]
            
            # Coagulation kernel
            K_ij = brownian_coagulation_kernel(
                d_i, d_j, temperature, viscosity, mean_free_path
            )
            
            # Collision probability in timestep
            # P = K * dt / V_cell (simplified)
            prob = K_ij * dt
            
            if prob > best_prob:
                best_prob = prob
                best_partner = neighbor
        
        neighbor = wp.hash_grid_query_next(query, grid)
    
    partner_indices[tid] = best_partner
    collision_probs[tid] = best_prob

@wp.kernel
def merge_coagulated_particles(
    diameters: wp.array(dtype=float),
    masses: wp.array(dtype=float),
    active: wp.array(dtype=int),
    partner_indices: wp.array(dtype=int),
    random_values: wp.array(dtype=float),
    collision_probs: wp.array(dtype=float),
):
    """Merge particles that collide (mass conserving)."""
    tid = wp.tid()
    
    if active[tid] == 0:
        return
    
    partner = partner_indices[tid]
    if partner < 0 or partner <= tid:  # Only process once per pair
        return
    
    # Probabilistic collision
    if random_values[tid] > collision_probs[tid]:
        return
    
    # Merge: larger particle absorbs smaller
    m_i = masses[tid]
    m_j = masses[partner]
    
    if m_i >= m_j:
        # Particle i absorbs j
        masses[tid] = m_i + m_j
        # New diameter from volume conservation
        # V = (4/3)*pi*r^3, so d_new = (d_i^3 + d_j^3)^(1/3)
        d_i = diameters[tid]
        d_j = diameters[partner]
        d_new = wp.pow(d_i*d_i*d_i + d_j*d_j*d_j, 1.0/3.0)
        diameters[tid] = d_new
        active[partner] = 0  # Deactivate absorbed particle
```

## Condensation and Evaporation

Mass transfer between gas phase and particles due to vapor pressure differences.

### Saturation and Kelvin Effect

```python
@wp.func
def saturation_vapor_pressure_water(temperature: float) -> float:
    """Saturation vapor pressure of water (Clausius-Clapeyron).
    
    Returns pressure in Pa.
    """
    T0 = 273.15  # Reference temperature (K)
    P0 = 611.2   # Reference pressure (Pa)
    L_v = 2.5e6  # Latent heat of vaporization (J/kg)
    R_v = 461.5  # Gas constant for water vapor (J/kg/K)
    
    return P0 * wp.exp((L_v / R_v) * (1.0/T0 - 1.0/temperature))

@wp.func
def kelvin_factor(
    diameter: float,
    surface_tension: float,
    molar_mass: float,
    density: float,
    temperature: float,
    gas_constant: float,  # from particula.util.constants.GAS_CONSTANT
) -> float:
    """Kelvin effect: increased vapor pressure over curved surface.
    
    Important for particles < 100 nm.
    gas_constant passed from particula.util.constants.GAS_CONSTANT.
    """
    exponent = (4.0 * surface_tension * molar_mass) / \
               (gas_constant * temperature * density * diameter)
    return wp.exp(exponent)

@wp.func
def equilibrium_saturation(
    diameter: float,
    dry_diameter: float,
    kappa: float,
    surface_tension: float,
    molar_mass: float,
    density: float,
    temperature: float,
) -> float:
    """Equilibrium saturation ratio including Raoult and Kelvin effects.
    
    Kappa-Kohler theory for hygroscopic growth.
    """
    # Kelvin term
    kelvin = kelvin_factor(diameter, surface_tension, molar_mass, 
                           density, temperature)
    
    # Raoult term (water activity from kappa)
    d_dry3 = dry_diameter * dry_diameter * dry_diameter
    d_wet3 = diameter * diameter * diameter
    a_w = (d_wet3 - d_dry3) / (d_wet3 - d_dry3 * (1.0 - kappa))
    
    return a_w * kelvin
```

### Condensation Growth Rate

```python
@wp.func
def mass_transfer_coefficient(
    diameter: float,
    diffusivity: float,
    mean_free_path: float,
    accommodation: float,
) -> float:
    """Fuchs-Sutugin correction for transition regime mass transfer."""
    kn = 2.0 * mean_free_path / diameter
    
    # Fuchs-Sutugin factor
    beta = (1.0 + kn) / \
           (1.0 + (4.0/(3.0*accommodation) + 0.377) * kn + \
            4.0/(3.0*accommodation) * kn * kn)
    
    return 2.0 * PI * diameter * diffusivity * beta

@wp.kernel
def compute_condensation_rates(
    diameters: wp.array(dtype=float),
    dry_diameters: wp.array(dtype=float),
    growth_rates: wp.array(dtype=float),
    temperature: float,
    saturation_ratio: float,
    vapor_diffusivity: float,
    mean_free_path: float,
    surface_tension: float,
    molar_mass: float,
    water_density: float,
    kappa: float,
    accommodation: float,
    gas_constant: float,  # from particula.util.constants.GAS_CONSTANT
):
    """Compute diameter growth rate dd/dt for each particle."""
    tid = wp.tid()
    
    d = diameters[tid]
    d_dry = dry_diameters[tid]
    
    # Equilibrium saturation for this particle
    S_eq = equilibrium_saturation(
        d, d_dry, kappa, surface_tension, molar_mass,
        water_density, temperature
    )
    
    # Supersaturation driving force
    delta_S = saturation_ratio - S_eq
    
    # Mass transfer coefficient
    k_m = mass_transfer_coefficient(
        d, vapor_diffusivity, mean_free_path, accommodation
    )
    
    # Saturation vapor pressure
    p_sat = saturation_vapor_pressure_water(temperature)
    
    # Growth rate: dd/dt = (k_m * M_w * p_sat * delta_S) / (rho_w * R * T * d)
    growth_rates[tid] = (k_m * molar_mass * p_sat * delta_S) / \
                        (water_density * gas_constant * temperature * d)

@wp.kernel
def integrate_condensation(
    diameters: wp.array(dtype=float),
    growth_rates: wp.array(dtype=float),
    dt: float,
    min_diameter: float,
):
    """Update particle diameters based on growth rates."""
    tid = wp.tid()
    
    d_new = diameters[tid] + growth_rates[tid] * dt
    
    # Prevent negative diameters (complete evaporation)
    if d_new < min_diameter:
        d_new = min_diameter
    
    diameters[tid] = d_new
```

## Brownian Diffusion

Random motion of particles due to molecular collisions.

### Brownian Displacement

```python
@wp.func
def brownian_step(
    diffusion_coeff: float,
    dt: float,
    random_x: float,
    random_y: float,
    random_z: float,
) -> wp.vec3:
    """Compute Brownian displacement for a timestep.
    
    Uses Box-Muller transform assumption that random values
    are already normally distributed.
    """
    # Standard deviation of displacement
    sigma = wp.sqrt(2.0 * diffusion_coeff * dt)
    
    return wp.vec3(
        sigma * random_x,
        sigma * random_y,
        sigma * random_z
    )

@wp.kernel
def apply_brownian_motion(
    positions: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    random_vectors: wp.array(dtype=wp.vec3),
    temperature: float,
    viscosity: float,
    mean_free_path: float,
    dt: float,
):
    """Apply Brownian diffusion to all particles."""
    tid = wp.tid()
    
    d = diameters[tid]
    D = diffusion_coefficient(d, temperature, viscosity, mean_free_path)
    
    # Random displacement
    rand = random_vectors[tid]
    displacement = brownian_step(D, dt, rand[0], rand[1], rand[2])
    
    positions[tid] = positions[tid] + displacement
```

## Complete Aerosol Simulation Loop

```python
def simulate_aerosol(
    n_particles: int = 10000,
    n_steps: int = 1000,
    dt: float = 0.001,
):
    """Complete aerosol dynamics simulation."""
    wp.init()
    
    # Environmental conditions
    temperature = 298.15  # K
    pressure = 101325.0   # Pa
    relative_humidity = 0.80
    
    # Air properties
    viscosity = 1.81e-5  # Pa·s
    mean_free_path = 6.8e-8  # m
    
    # Water properties
    surface_tension = 0.072  # N/m
    molar_mass = 0.018  # kg/mol
    water_density = 1000.0  # kg/m³
    vapor_diffusivity = 2.5e-5  # m²/s
    
    # Initialize particle population (log-normal distribution)
    geometric_mean = 100e-9  # 100 nm
    geometric_std = 1.8
    log_diameters = np.random.normal(
        np.log(geometric_mean),
        np.log(geometric_std),
        n_particles
    )
    initial_diameters = np.exp(log_diameters)
    
    # Warp arrays
    diameters = wp.array(initial_diameters, dtype=float, device="cuda")
    dry_diameters = wp.array(initial_diameters, dtype=float, device="cuda")
    positions = wp.array(
        np.random.uniform(-0.5, 0.5, (n_particles, 3)),
        dtype=wp.vec3, device="cuda"
    )
    concentrations = wp.array(
        np.ones(n_particles) * 1e6,  # #/m³
        dtype=float, device="cuda"
    )
    growth_rates = wp.zeros(n_particles, dtype=float, device="cuda")
    coag_rates = wp.zeros(n_particles, dtype=float, device="cuda")
    
    # Saturation ratio from relative humidity
    saturation_ratio = relative_humidity
    
    # Simulation loop
    for step in range(n_steps):
        # 1. Compute condensation growth
        wp.launch(
            compute_condensation_rates,
            dim=n_particles,
            inputs=[
                diameters, dry_diameters, growth_rates,
                temperature, saturation_ratio, vapor_diffusivity,
                mean_free_path, surface_tension, molar_mass,
                water_density, 0.3,  # kappa for ammonium sulfate
                1.0,  # accommodation coefficient
            ]
        )
        
        wp.launch(
            integrate_condensation,
            dim=n_particles,
            inputs=[diameters, growth_rates, dt, 1e-9]
        )
        
        # 2. Compute coagulation rates
        wp.launch(
            compute_coagulation_rates,
            dim=n_particles,
            inputs=[
                diameters, concentrations, coag_rates,
                temperature, viscosity, mean_free_path, n_particles
            ]
        )
        
        # 3. Apply Brownian diffusion
        random_vecs = wp.array(
            np.random.randn(n_particles, 3),
            dtype=wp.vec3, device="cuda"
        )
        wp.launch(
            apply_brownian_motion,
            dim=n_particles,
            inputs=[
                positions, diameters, random_vecs,
                temperature, viscosity, mean_free_path, dt
            ]
        )
        
        wp.synchronize()
        
        if step % 100 == 0:
            d_mean = np.mean(diameters.numpy())
            print(f"Step {step}: mean diameter = {d_mean*1e9:.1f} nm")
    
    return {
        "diameters": diameters.numpy(),
        "positions": positions.numpy(),
        "concentrations": concentrations.numpy(),
    }
```

## Size Distribution Moment Calculations

```python
@wp.kernel
def compute_distribution_moments(
    diameters: wp.array(dtype=float),
    concentrations: wp.array(dtype=float),
    moments: wp.array(dtype=float),  # [M0, M1, M2, M3, M6]
):
    """Compute moments of the size distribution.
    
    M0 = total number concentration
    M1 = mean diameter weighted by number
    M2 = proportional to surface area
    M3 = proportional to volume/mass
    M6 = needed for optical properties
    """
    tid = wp.tid()
    
    d = diameters[tid]
    n = concentrations[tid]
    
    d2 = d * d
    d3 = d2 * d
    d6 = d3 * d3
    
    wp.atomic_add(moments, 0, n)           # M0
    wp.atomic_add(moments, 1, n * d)       # M1
    wp.atomic_add(moments, 2, n * d2)      # M2
    wp.atomic_add(moments, 3, n * d3)      # M3
    wp.atomic_add(moments, 4, n * d6)      # M6
```

## Performance Optimization

### Use Spatial Hashing for O(n) Coagulation

```python
def create_particle_grid(positions, cell_size):
    """Create spatial hash grid for efficient neighbor queries."""
    grid = wp.HashGrid(dim_x=64, dim_y=64, dim_z=64)
    
    wp.launch(
        build_particle_grid,
        dim=positions.shape[0],
        inputs=[positions, grid.id, cell_size]
    )
    
    return grid

@wp.kernel
def build_particle_grid(
    positions: wp.array(dtype=wp.vec3),
    grid: wp.uint64,
    cell_size: float,
):
    tid = wp.tid()
    wp.hash_grid_point_id(grid, tid, positions[tid])
```

## See Also

- [Kernels and Function Inlining](../kernels.md) - Kernel optimization
- [Data Structures](../datastructures.md) - Particle data types
- [Particle Interactions](./geometry.md) - Wall loss, collisions
