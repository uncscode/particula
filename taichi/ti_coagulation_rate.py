"""
Taichi implementation of the coagulation gain rate calculation.
"""
import timeit
import taichi as ti
import numpy as np
import particula as par

ti.init(arch=ti.cpu)  # or ti.cpu

size = 250  # Number of kernel size bins

# radius_type = ti.MatrixField(
#     dtype=ti.f64, shape=(size,)
# )
# concentration_type = ti.ndarray(
#     dtype=ti.f64, shape=(size,)
# )
# kernel_type = ti.ndarray(
#     dtype=ti.f64, shape=(size, size)
# )
# gain_rate_type = ti.ndarray(
#     dtype=ti.f64, shape=(size,)
# )
# output_type = ti.ndarray(
#     dtype=ti.f64, shape=(size,)
# )


@ti.kernel
def get_coagulation_gain_rate_continuous_taichi(
    radius: ti.types.ndarray(),
    concentration: ti.types.ndarray(),
    kernel: ti.types.ndarray(),
    gain_rate: ti.types.ndarray(),
) -> ti.types.ndarray():
    """
    Calculate the coagulation gain rate via discrete trapezoidal integration.

    gain_rate[i] = ∑_j K[i,j] · c[i] · c[j] · Δr_j

    Args:
        radius: Particle radius array.
        concentration: Particle concentration array.
        kernel: Coagulation kernel matrix.
        gain_rate: Output array to write gain rates into.
    """
    n = radius.shape[0]
    for i in range(n):
        acc = 0.0
        for j in range(n - 1):
            dr = radius[j + 1] - radius[j]
            # trapezoid: ½·[f(j) + f(j+1)]·Δr
            acc += (
                0.5
                * (
                    kernel[i, j] * concentration[i] * concentration[j]
                    + kernel[i, j + 1]
                    * concentration[i]
                    * concentration[j + 1]
                )
                * dr
            )
        gain_rate[i] = acc
    return gain_rate


if __name__ == "__main__":
    # --- example usage ---
    # Create fine scale radius bins on a logarithmic scale from 1 nm to 10 μm
    radius_bins = np.logspace(start=-9, stop=-4, num=250)  # m (1 nm to 10 μm)

    # Calculate the mass for each particle size bin assuming a density of 1 g/cm^3 (1000 kg/m^3)
    mass_bins = 4 / 3 * np.pi * (radius_bins) ** 3 * 1e3  # kg

    # Generate a lognormal particle size distribution
    # This distribution is characterized by a mode (most probable size) of 100 nm,
    # a geometric standard deviation of 1.4, and a total number concentration of 10000 particles per cm^3.
    concentration_lognormal_0 = par.particles.get_lognormal_pmf_distribution(
        x_values=radius_bins,
        mode=np.array(100e-9),  # Mode of the distribution (100 nm)
        geometric_standard_deviation=np.array(1.4),  # Geometric standard deviation
        number_of_particles=np.array(
            1e6 * 1e6  # Total concentration (10000 cm^-3 converted to m^-3)
        ),
    )

    # Calculate the coagulation kernel
    kernel = par.dynamics.get_brownian_kernel_via_system_state(
        particle_radius=radius_bins,
        particle_mass=mass_bins,
        temperature=293.15,  # Temperature in Kelvin
        pressure=101325,  # Pressure in Pascals (1 atm)
        alpha_collision_efficiency=1.0,  # Assume perfect collision efficiency
    )

    out = np.zeros_like(concentration_lognormal_0, dtype=np.float64)

    # to taichi
    radius_bins = np.asarray(radius_bins, dtype=np.float32)
    concentration_lognormal_0 = np.asarray(
        concentration_lognormal_0, dtype=np.float32
    )
    kernel = np.asarray(kernel, dtype=np.float32)
    out = np.asarray(out, dtype=np.float32)
    get_coagulation_gain_rate_continuous_taichi(
        radius_bins,
        concentration_lognormal_0,
        kernel,
        out)
    print("gain_rate:", out)
