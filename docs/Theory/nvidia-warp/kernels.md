# Kernels and Function Inlining for Particle Simulation

## Introduction

Kernels are the fundamental building blocks of parallel computation in Warp. For particula, kernels enable massively parallel computation of aerosol dynamics where each particle can be processed independently.

## Kernel Basics

### Defining a Kernel

Kernels are Python functions decorated with `@wp.kernel`:

```python
import warp as wp

@wp.kernel
def compute_particle_masses(
    diameters: wp.array(dtype=float),
    densities: wp.array(dtype=float),
    masses: wp.array(dtype=float),
):
    """Compute particle mass from diameter and density."""
    tid = wp.tid()
    
    # Volume of sphere: (4/3) * pi * r^3
    radius = diameters[tid] * 0.5
    volume = (4.0 / 3.0) * 3.14159265359 * radius * radius * radius
    
    masses[tid] = densities[tid] * volume
```

### Key Characteristics

1. **Thread Index**: Use `wp.tid()` to get the current particle's index
2. **Type Annotations**: All parameters must have Warp type annotations
3. **Parallel Execution**: Each particle is processed independently
4. **No Return Values**: Kernels write results to output arrays

### Launching Kernels

```python
# Create particle arrays
n_particles = 10000
diameters = wp.array(np.ones(n_particles) * 1e-6, dtype=float)  # 1 micron
densities = wp.array(np.ones(n_particles) * 1000.0, dtype=float)  # kg/m³
masses = wp.zeros(n_particles, dtype=float)

# Launch kernel - one thread per particle
wp.launch(compute_particle_masses, dim=n_particles, inputs=[diameters, densities, masses])
```

## Function Inlining

### Warp Functions

Functions decorated with `@wp.func` can be called from kernels and are automatically inlined during compilation for optimal performance.

```python
@wp.func
def knudsen_number(diameter: float, mean_free_path: float) -> float:
    """Calculate Knudsen number for a particle."""
    return 2.0 * mean_free_path / diameter

@wp.func
def cunningham_correction(knudsen: float) -> float:
    """Cunningham slip correction factor."""
    # Cc = 1 + Kn * (A1 + A2 * exp(-A3/Kn))
    A1 = 1.257
    A2 = 0.400
    A3 = 0.55
    return 1.0 + knudsen * (A1 + A2 * wp.exp(-A3 / knudsen))

@wp.kernel
def compute_slip_corrections(
    diameters: wp.array(dtype=float),
    mean_free_path: float,
    corrections: wp.array(dtype=float),
):
    """Compute Cunningham slip correction for each particle."""
    tid = wp.tid()
    kn = knudsen_number(diameters[tid], mean_free_path)
    corrections[tid] = cunningham_correction(kn)
```

### Benefits of Inlining

1. **Zero Overhead**: Function calls are eliminated at compile time
2. **Optimization**: Compiler can optimize across function boundaries
3. **Code Reuse**: Write modular physics functions without performance penalty
4. **Type Safety**: Full type checking at compile time

## Advanced Kernel Patterns

### Multi-Dimensional Kernels

For particle-particle interactions like coagulation:

```python
@wp.kernel
def compute_coagulation_kernel_matrix(
    diameters: wp.array(dtype=float),
    kernel_matrix: wp.array2d(dtype=float),
    temperature: float,
    viscosity: float,
):
    """Compute coagulation kernel K(i,j) for all particle pairs."""
    i, j = wp.tid()
    
    d_i = diameters[i]
    d_j = diameters[j]
    
    # Brownian coagulation kernel (simplified)
    k_B = 1.380649e-23  # Boltzmann constant
    coag_kernel = (2.0 * k_B * temperature / (3.0 * viscosity)) * \
                  (d_i + d_j) * (1.0/d_i + 1.0/d_j)
    
    kernel_matrix[i, j] = coag_kernel

# Launch with 2D dimensions
wp.launch(compute_coagulation_kernel_matrix, 
          dim=(n_particles, n_particles), 
          inputs=[diameters, kernel_matrix, temperature, viscosity])
```

### Conditional Logic

```python
@wp.kernel
def classify_particles(
    diameters: wp.array(dtype=float),
    classifications: wp.array(dtype=int),
):
    """Classify particles by size regime."""
    tid = wp.tid()
    d = diameters[tid]
    
    # PM2.5: < 2.5 microns, PM10: < 10 microns
    if d < 2.5e-6:
        classifications[tid] = 0  # PM2.5 (fine)
    elif d < 10.0e-6:
        classifications[tid] = 1  # PM10 (coarse)
    else:
        classifications[tid] = 2  # Large particles
```

### Atomic Operations

For operations that need synchronization across particles (e.g., total mass):

```python
@wp.kernel
def compute_total_mass(
    masses: wp.array(dtype=float),
    concentrations: wp.array(dtype=float),
    total_mass: wp.array(dtype=float),
):
    """Compute total aerosol mass concentration."""
    tid = wp.tid()
    mass_conc = masses[tid] * concentrations[tid]
    wp.atomic_add(total_mass, 0, mass_conc)
```

## Performance Considerations

### Best Practices

1. **Minimize Memory Access**: Reuse loaded particle properties
2. **Avoid Divergence**: Keep conditional branches minimal
3. **Use Local Variables**: Store frequently accessed data in registers
4. **Coalesce Memory Access**: Access particle arrays in contiguous patterns

### Example: Optimized Condensation Kernel

```python
@wp.func
def saturation_vapor_pressure(temperature: float) -> float:
    """Clausius-Clapeyron equation for water vapor."""
    # Reference: T0 = 273.15 K, P0 = 611.2 Pa
    L_v = 2.5e6  # Latent heat of vaporization (J/kg)
    R_v = 461.5  # Gas constant for water vapor (J/kg/K)
    T0 = 273.15
    P0 = 611.2
    
    return P0 * wp.exp((L_v / R_v) * (1.0/T0 - 1.0/temperature))

@wp.func
def kelvin_effect(diameter: float, surface_tension: float, 
                  molar_mass: float, density: float, 
                  temperature: float) -> float:
    """Kelvin effect correction for small droplets."""
    R = 8.314  # Universal gas constant
    exponent = (4.0 * surface_tension * molar_mass) / \
               (R * temperature * density * diameter)
    return wp.exp(exponent)

@wp.kernel
def compute_condensation_rates(
    diameters: wp.array(dtype=float),
    temperatures: wp.array(dtype=float),
    vapor_pressures: wp.array(dtype=float),
    growth_rates: wp.array(dtype=float),
    surface_tension: float,
    molar_mass: float,
    density: float,
):
    """Compute condensation growth rate for each particle."""
    tid = wp.tid()
    
    # Load particle properties once
    d = diameters[tid]
    T = temperatures[tid]
    p_vapor = vapor_pressures[tid]
    
    # Compute equilibrium vapor pressure with Kelvin effect
    p_sat = saturation_vapor_pressure(T)
    kelvin = kelvin_effect(d, surface_tension, molar_mass, density, T)
    p_eq = p_sat * kelvin
    
    # Supersaturation
    S = p_vapor / p_eq - 1.0
    
    # Growth rate (simplified Maxwell equation)
    D_v = 2.5e-5  # Diffusion coefficient of water vapor (m²/s)
    growth_rates[tid] = (2.0 * D_v * molar_mass * p_eq * S) / \
                        (density * 8.314 * T * d)
```

## Debugging Kernels

### Print Statements

```python
@wp.kernel
def debug_particle_kernel(
    diameters: wp.array(dtype=float),
    masses: wp.array(dtype=float),
):
    tid = wp.tid()
    
    if tid == 0:  # Print only from first thread
        wp.printf("First particle: d=%e m, m=%e kg\n", 
                  diameters[0], masses[0])
```

### Bounds Checking

Warp automatically checks array bounds in debug mode:

```python
wp.config.verify_cuda = True  # Enable bounds checking
```

## Common Patterns for Aerosol Physics

### Particle Size Distribution Moments

```python
@wp.kernel
def compute_moments(
    diameters: wp.array(dtype=float),
    concentrations: wp.array(dtype=float),
    moments: wp.array(dtype=float),  # [M0, M1, M2, M3]
):
    """Compute size distribution moments."""
    tid = wp.tid()
    
    d = diameters[tid]
    n = concentrations[tid]
    
    # M_k = sum(n_i * d_i^k)
    wp.atomic_add(moments, 0, n)              # M0 (number)
    wp.atomic_add(moments, 1, n * d)          # M1 (mean diameter)
    wp.atomic_add(moments, 2, n * d * d)      # M2 (surface area)
    wp.atomic_add(moments, 3, n * d * d * d)  # M3 (volume/mass)
```

### Diffusion Coefficient Calculation

```python
@wp.func
def stokes_einstein_diffusion(
    diameter: float,
    temperature: float,
    viscosity: float,
    slip_correction: float,
) -> float:
    """Stokes-Einstein diffusion coefficient with slip correction."""
    k_B = 1.380649e-23  # Boltzmann constant
    return (k_B * temperature * slip_correction) / \
           (3.0 * 3.14159265359 * viscosity * diameter)

@wp.kernel
def compute_diffusion_coefficients(
    diameters: wp.array(dtype=float),
    slip_corrections: wp.array(dtype=float),
    diffusion_coeffs: wp.array(dtype=float),
    temperature: float,
    viscosity: float,
):
    """Compute diffusion coefficient for each particle."""
    tid = wp.tid()
    diffusion_coeffs[tid] = stokes_einstein_diffusion(
        diameters[tid], temperature, viscosity, slip_corrections[tid]
    )
```

## See Also

- [Data Structures](./datastructures.md) - Particle and aerosol data types
- [Aerosol Dynamics Example](./examples/fluids.md) - Coagulation, condensation
- [Particle Interactions Example](./examples/geometry.md) - Collisions, wall loss
