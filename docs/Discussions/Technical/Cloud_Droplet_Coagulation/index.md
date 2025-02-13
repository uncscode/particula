# Cloud Droplet Coagulation

In this folder, we discuss the implementation of the geometric collision kernel for cloud droplets as described in Part II by Ayala et al. (2008). Part I provides a detailed explanation of the direct numerical simulations. Where as Part II is the parameterization of the collision kernel for cloud droplets in turbulent flows. The implementation involves calculating the geometric collision rate of sedimenting droplets based on the turbulent flow properties and droplet characteristics.

Ayala, O., Rosa, B., Wang, L. P., & Grabowski, W. W. (2008). Effects of turbulence on the geometric collision rate of sedimenting droplets. Part 1. Results from direct numerical simulation. New Journal of Physics, 10. https://doi.org/10.1088/1367-2630/10/7/075015

Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on the geometric collision rate of sedimenting droplets. Part 2. Theory and parameterization. New Journal of Physics, 10. https://doi.org/10.1088/1367-2630/10/7/075016

## Geometric Collision Kernel Γ₁₂ 

The geometric collision kernel from the paper is outlined in [Ayala et al. (2008)](Droplet_Coagulation_Kernel_Ayala2008.md).

## Validations

We validate our implementation of the geometric collision kernel against the results from Ayala et al. (2008) via jupyter notebooks. The notebooks cover comparison graphs and tables from the original paper and Direct Numerical Simulations (DNS) results.

- [DNS Fluid and Particle Properties](DNS_Fluid_and_Particle_Properties_Comparison.ipynb)
- [DNS Horizontal Velocity](DNS_Horizontal_Velocity_Comparison.ipynb)
- [DNS Radial Relative Velocity](DNS_Radial_Distribution_Comparison.ipynb)
- [DNS Radial Distribution](DNS_Radial_Distribution_Comparison.ipynb)
- [DNS Kernel Comparison](DNS_Kernel_Comparison.ipynb)