# NVIDIA Warp for Particle Simulation

## Overview

NVIDIA Warp is a Python framework for writing high-performance simulation code. Warp takes regular Python functions and JIT compiles them to efficient kernel code that can run on the CPU or GPU. For particula, Warp enables GPU-accelerated aerosol dynamics simulations including coagulation, condensation, and particle transport.

## Key Features

- **Python-First**: Write simulation code in Python with decorators
- **GPU Acceleration**: Automatic compilation to CUDA kernels
- **Differentiable**: Built-in support for automatic differentiation
- **Efficient**: Zero-copy interop with NumPy and other frameworks

## Core Concepts

### 1. Kernels

Warp kernels are Python functions decorated with `@wp.kernel` that execute in parallel across particles. They are compiled to native code and can run on CPU or GPU.

```python
import warp as wp

@wp.kernel
def update_particle_positions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
):
    tid = wp.tid()
    positions[tid] = positions[tid] + velocities[tid] * dt
```

### 2. Functions

Warp functions are reusable code blocks decorated with `@wp.func` that can be called from kernels. They are automatically inlined during compilation.

```python
@wp.func
def brownian_displacement(
    diffusion_coeff: float,
    dt: float,
    random_val: wp.vec3,
) -> wp.vec3:
    """Compute Brownian motion displacement for a particle."""
    sigma = wp.sqrt(2.0 * diffusion_coeff * dt)
    return random_val * sigma
```

### 3. Data Structures

Warp provides efficient data structures optimized for parallel computation:
- **Arrays**: 1D, 2D, 3D typed arrays for particle properties
- **Vectors**: `wp.vec3` for positions, velocities
- **Matrices**: `wp.mat33` for rotation, coagulation kernels
- **Custom Structs**: Define `Particle`, `Aerosol` types

## Getting Started

### Installation

```bash
pip install warp-lang
```

### Basic Usage

```python
import warp as wp
import numpy as np

# Initialize Warp
wp.init()

# Create particle arrays on GPU
n_particles = 10000
positions = wp.array(np.random.randn(n_particles, 3) * 1e-6, dtype=wp.vec3, device="cuda")
velocities = wp.zeros(n_particles, dtype=wp.vec3, device="cuda")

# Launch kernel
dt = 0.001  # 1 ms timestep
wp.launch(update_particle_positions, dim=n_particles, inputs=[positions, velocities, dt])

# Synchronize and get results
wp.synchronize()
result = positions.numpy()
```

## Resources

- [NVIDIA Warp GitHub](https://github.com/NVIDIA/warp)
- [NVIDIA Warp Documentation](https://nvidia.github.io/warp/)
- [particula Documentation](https://uncscode.github.io/particula/)

## Table of Contents

- [Kernels and Function Inlining](./kernels.md) - Writing parallel particle computations
- [Data Structures](./datastructures.md) - Particle and aerosol data types
- [Examples](./examples/)
  - [Aerosol Dynamics](./examples/fluids.md) - Coagulation, condensation, diffusion
  - [Particle Interactions](./examples/geometry.md) - Collisions, wall losses
