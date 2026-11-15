# Optical calculations

Particula's development optics API calculates spherical-particle scattering on
NumPy CPU, Warp CPU and CUDA. Import numerical CPU functions from
`particula.optics`; import device workspaces explicitly from
`particula.optics.gpu`. No PyMieScatt installation is needed.

Start with the executed [spectra, polarization and cloud example](../../examples/optics/sphere_optics.py).
This feature supplies optical properties for later radiation and microphysics
models. It does not update particle mass, temperature, gas or environment state.

## Physical scope and units

The model is Lorenz–Mie scattering by homogeneous, nonmagnetic spheres in a
nonabsorbing medium. Population properties assume independent, incoherent
scattering. Core-shell particles, nonspherical shapes, dependent scattering,
material mixing laws and multiple-scattering radiative transfer are outside
this implementation.

Use radius and vacuum wavelength in metres, angles in radians, and absolute
particle refractive index `n + 1j*k` with `n > 0` and `k >= 0`. The medium index
is positive and real. Normalize exactly once:

- Relative index: particle index / medium index.
- Size parameter: `x = 2*pi*radius*medium_index/wavelength`.
- Efficiencies: dimensionless; cross sections: m².
- Differential cross sections: m²/sr; bulk coefficients: m⁻¹.
- Optical depth and direct transmission: dimensionless.

The supported size-parameter domain is `0 <= x <= 10000`. Finite input
validation and bounded numerical convergence are separate: a numerically
unresolved calculation raises an error or sets a device failure status.
The [validation record](../../contribute/validation/optics/reference-corpus.md)
identifies tested cases, reference discrepancies and acceptance criteria.

## Particle spectra and retained absorption

```python
import numpy as np
from particula.optics import get_mie_efficiencies, get_cross_sections

wavelength = np.linspace(400e-9, 1000e-9, 121)
radius = 200e-9
# Illustrative constant index, not measured material spectral data.
q = get_mie_efficiencies(radius, wavelength, 1.5 + 0.02j)
cross = get_cross_sections(q, radius)
particle_absorption_m2 = cross.absorption
```

The efficiency result contains extinction, scattering, absorption, asymmetry
`g`, radar-convention backscatter, radiation pressure and a scattering-validity
mask. Radar backscatter is an area, not a differential area per steradian.
Cross sections multiply the corresponding efficiencies by `pi*radius**2`.

Inputs follow NumPy broadcasting. For a spectrum for each particle, use
`radius[:, None]` and `wavelength[None, :]`. Coefficients append an order axis;
angular amplitudes append an angle axis; Mueller matrices append `(4, 4)`.
Zero radius and index matching give zero scattering. Undefined normalized
quantities return zero with an explicit validity mask.

## Angular scattering and polarization

`get_mie_coefficients(relative_index, x)` returns reusable coefficients.
`get_scattering_amplitudes(coefficients, angles)` returns perpendicular `S1`
and parallel `S2`. The scalar-angle case retains a length-one angle axis.

`get_differential_cross_sections(S1, S2, wavelength, medium_index)` returns
perpendicular, parallel and unpolarized differential cross sections.
`get_mueller_matrix(S1, S2)` gives the raw sphere matrix;
`get_differential_mueller_matrix` converts it to m²/sr, and `get_phase_matrix`
divides by total scattering so element 11 integrates to one over solid angle.
Always use the wavelength, medium and scattering cross section of the same
particle calculation.

The scattering-plane Stokes convention is:

- `I = |E_parallel|² + |E_perpendicular|²`.
- `Q = |E_parallel|² - |E_perpendicular|²`.
- `U = 2 Re(E_parallel E_perpendicular*)`.
- `V = -2 Im(E_parallel E_perpendicular*)`.

`rotate_stokes(S, angle)` rotates the reference axes, with
`Q' = Q*cos(2*angle) + U*sin(2*angle)` and
`U' = -Q*sin(2*angle) + U*cos(2*angle)`.
`apply_mueller_matrix(M, S, rotation_in, rotation_out)` computes
`R(rotation_out) M R(rotation_in) S`. Supply rotations explicitly when
incident/output reference frames differ from the scattering plane. These
linear helpers accept signed Stokes combinations; they do not certify that an
arbitrary supplied vector represents physically realizable light.

## Population coefficients and path geometry

`get_population_weights` converts a final particle axis into number-density
weights:

| Distribution | Input | Required information |
| --- | --- | --- |
| `discrete` | Bin number density [m⁻³] | None |
| `particle_resolved` | Represented particle count | Representative volume [m³] |
| `continuous_pdf` | Radius density dN/dr [m⁻⁴] | Increasing radius grid [m] |

Use the exact distribution names in the API. PDF weights apply trapezoidal
quadrature to the product of number density and optical cross section.
Numerical GPU PDF integration does not imply support for native GPU PDF
microphysics simulations.

`get_population_optics(cross_sections, asymmetry, weights, particle_axis=-1)`
sums particle cross sections times density. Bulk single-scattering albedo is
summed scattering / summed extinction. Bulk `g` is weighted by scattering,
not by particle count alone. `get_population_mueller_matrix` sums differential
matrices in a common reference frame; amplitudes are never coherently summed.

`get_column_optics(extinction, path_length, layer_axis=-1)` multiplies each
layer's coefficient by its explicit length and sums optical depth. It returns
layer depth, total depth and `exp(-tau)`. Representative particle volume does
not define a light path. Direct transmission describes the unscattered beam;
it is not total flux transmission through a multiply scattering cloud.
`get_mass_specific_opacity` additionally requires explicit mass density
[kg/m³], returning m²/kg with a validity mask for zero-density cases.

## Device calculations

Run the [CUDA optics example](../../examples/optics/cuda_optics.py) for explicit
transfers, timed particle batches and a streamed cloud spectrum. It requires
native CUDA and verifies results against CPU calculations.

Upload inputs explicitly to the chosen Warp device. `prepare_mie` accepts
contiguous 1D `wp.vec2d` relative indices (real, imaginary) and `wp.float64`
size parameters. Optional shared angles are a 1D float64 array. It retains
output buffers and precompiled commands; preparation does not advance the
calculation. Call `enqueue()` to calculate and `check()` to synchronize and
reject failed results. Reading `.numpy()` is an explicit host download.

`prepare_population_mie` accepts `(groups, particles)` device arrays of relative
index, size parameter, radius and number-density weights. Groups may represent
box/wavelength pairs. It streams particle chunks and retains only bulk optical
coefficients and optional bulk differential Mueller matrices. Scratch depends
on particle-tile size, coefficient capacity and requested angle count, rather
than total population size. Per-particle Mie workspaces retain requested
particle outputs; use them when individual absorption is needed.

The direct kernels in `particula.optics.gpu.mie`, `angular`, `cross_sections`,
`population` and `column` expose caller-owned arrays. In particular,
`cross_sections_kernel` converts particle efficiencies and radii to retained
extinction/scattering/absorption/backscatter/radiation-pressure areas.
Follow each kernel's shapes, dtypes, status codes and launch dimensions.
Direct launches do not perform Python metadata admission. Failed outputs must
not be consumed, and a failed kernel can leave partial output values.

Input/output storage and stream ordering remain the caller's responsibility.
Update inputs in place on the prepared stream, then enqueue again. Rebuild
workspaces for changed shapes, devices or capacities. Do not concurrently
launch the same workspace on different streams: its scratch is shared.
There is no hidden transfer, synchronization in enqueue, or CPU fallback.

## Later thermal and evaporation feedback

Keep optical evaluation separate from a custom state-changing process:

1. Read current radii, composition-dependent indices and density weights.
2. Calculate individual absorption and bulk optical fields.
3. Obtain a radiation field from a specified source/transport model.
4. Integrate particle absorption against spectral irradiance for absorbed power.
5. Solve the energy and mass-transfer balances, including emission, heat
   exchange and latent heat, before updating particle/environment state.

The optical coefficients alone do not determine temperature or evaporation.
Feedback must refresh optics as size/composition changes and must enforce the
owning process's conservation, volume and mutation contracts.

## Scientific basis

The coefficient, angular and polarization conventions follow Bohren and
Huffman, *Absorption and Scattering of Light by Small Particles* (1983),
[chapter 4](https://doi.org/10.1002/9783527618156). Reference validation combines
stored PyMieScatt observations with independent Bessel/polynomial calculations,
energy identities and CPU/CUDA comparisons. Known reference discrepancies are
preserved and documented rather than reproduced as physical behavior.
