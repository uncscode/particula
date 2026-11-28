# Optical calculations

Particula's development optics API calculates spherical-particle scattering on
NumPy CPU, Warp CPU and CUDA. Import numerical CPU functions from
`particula.optics`; import device workspaces explicitly from
`particula.optics.gpu`. No PyMieScatt installation is needed.

Start with the executed [spectra, polarization and cloud example](../../studies/validation/material-properties/sphere-optics/sphere_optics.py).
Optical calculations supply properties for observation and feedback. A
heating runnable uses particle absorption to advance temperature and water
transfer in a separate step.

## Physical scope and units

The model is Lorenz–Mie scattering by homogeneous, nonmagnetic spheres in a
nonabsorbing medium. Population properties assume independent, incoherent
scattering. This sphere model describes individual scattering events. The CPU
shortwave solver below uses layer extinction, scattering, and phase moments
to propagate light through multiple events. Users can also supply bulk
optical properties from external measurements or particle models.

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
The [validation record](../../studies/validation/material-properties/sphere-optics/index.md)
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

For native CPU particle state, `population_optics` uses each box's current
volume and distribution. An independent extinction check is the sum of
cross section times integrated number density. For a discrete sample at a
550 nm wavelength, compare the bulk extinction in m⁻¹ as follows.

```python
import numpy as np
from particula.optics import OpticalConfig, particle_optics, population_optics
from particula.particles import create_particle_data_from_radii

sample = create_particle_data_from_radii(
    radii=[50e-9, 120e-9], concentration=[4.0, 8.0], volume=0.25,
    distribution_type="discrete",
)
config = OpticalConfig([550e-9], particle_index=1.5 + 0.02j)
_, cross = particle_optics(sample, config)
expected = np.sum(cross.extinction[0, 0] * np.array([4.0, 8.0]))
np.testing.assert_allclose(population_optics(sample, config).extinction[0, 0],
                           expected, atol=0)
```

For a native radius PDF, each contribution also includes its trapezoidal
radius-node width in metres. An optical path length enters only when
converting bulk extinction to optical depth.
[Particle units and quadrature](../migration/v0.4.0/index.md#check-integration-mass-and-diameter-spectra)

For independent measurement arrays, `get_population_weights` retains its
input conventions. Choose the input that matches the measurement:

| Distribution | Input | Required information |
| --- | --- | --- |
| `discrete` | Bin number density [m⁻³] | None |
| `particle_resolved` | Represented particle count | Representative volume [m³] |
| `continuous_pdf` | Radius density dn/dr [m⁻⁴] | Increasing radius grid [m] |

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

## Solar flux through a parcel or column

Use `earth_reference` in `particula.optics.solar` to set an ideal, cloud-free
atmosphere and incident solar spectrum for a latitude, east longitude, UTC
timestamp, and descending interface heights in metres. Its upper face
receives extraterrestrial sunlight. Supply the atmosphere above the aerosol
when requesting a surface or top-of-atmosphere flux. The installed
reference includes the TSIS-1 Sun and a fixed US standard atmospheric
profile. Actual weather and surface type are user inputs.

A native particle population can supply one or more slab optical depths
through `SpectralColumn.from_population`. Map boxes to layers explicitly and
combine the aerosol and atmospheric optical depths with `plus`. A bounded
parcel calculation also accepts a measured or prescribed `SolarInput` at
its upper face. Direct-normal irradiance and diffuse horizontal irradiance
are distinct inputs. Report the height at which the light was measured.

Call `solve_shortwave(column, sun, surface)` for the upwelling, diffuse-down
and direct-down spectral fluxes at each interface. Integrate each result with
its wavelength measure. The reference ships a reduced quadrature for broad
bands and a full REPTRAN quadrature for finer band averages. Increasing
angular streams tests the angular solution at a fixed spectral grid. A
satellite band additionally requires resolved spectral support across its
response and the sensor's viewing direction. The surface options include
Lambertian ocean and dusty-ground references, Ross–Li land reflection, and
wind-dependent Cox–Munk ocean glint.

For a sounding or reanalysis profile, supply `pressure_pa` and
`gas_optical_depth` to `earth_reference` with a descriptive `profile_name`.
You can compute the latter with `profile_gas_optical_depth` using interface
temperatures in K, gas mole fractions in mol/mol and layer absorption cross
sections in m² per molecule. Supplying pressure alone changes Rayleigh
scattering and retains the reference gas absorption. Specify the lowest
interface at the site's elevation. The reference spans −500 to 120,000 m,
with the lowest standard layer extrapolated below sea level.

For a cloud-free land pixel, `particula.data.mcd43.read_mcd43_surface` reads
matching local Collection 6.1 MCD43A1 and A2 tiles using optional `pyhdf`.
Supply the row, column and an explicit MODIS band for each wavelength. The
reader accepts full inversions (band quality 0 or 1) and records the tile
and retrieval window. For ocean, `ocean_surface` accepts an optional isotropic
water-leaving spectral radiance in W m⁻² sr⁻¹ m⁻¹. Its wind-dependent option
models glint and omits whitecaps. [MODIS band quality definitions](https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MCD43A2)

NASA POWER supplies optional hourly surface irradiance. CAMS also offers
historical global clear-sky series and regional all-sky series. Preserve
the requested time interval, sky condition, direct-normal convention,
and units. For CAMS irradiation reported in Wh/m², divide by the interval
in hours to obtain mean W/m². The user-supplied spectral shape and band
fractions then specify how the measured energy enters the calculation.
[CAMS coverage and access](https://ads.atmosphere.copernicus.eu/datasets/cams-solar-radiation-timeseries?tab=overview)

The [parcel over two surfaces](../../studies/examples/coupled-populations/radiation/parcel_surfaces.py)
compares matched aerosol-present and aerosol-free columns. The
[layer altitude example](../../studies/examples/coupled-populations/radiation/layer_altitude.py)
connects a site comparison to the optional real-data workflow.
See [spectral transfer](../../theory/optics/spectral-transport.md) for the
energy budget, [surface and sensor geometry](../../theory/optics/surface-and-sensor.md)
for radiance units, and [solver benchmarks](../../studies/validation/spectral-radiation/benchmark.md)
for accuracy and low-Sun limits.

### Compare several columns on a Warp device

To isolate the effect of the ground, use the same atmospheric column and
sunlight for two calculations and change only the surface. Here a clear
column at 20° N extends from sea level to 120 km. Prepare the optical
inputs once in Python before sending the two cases to the device.

```python
from datetime import UTC, datetime

from particula.optics import dusty_ground, earth_reference, ocean_surface

earth = earth_reference(
    20.0, 0.0, datetime(2022, 6, 21, 12, tzinfo=UTC), [120_000, 1_000, 0]
)
wave = earth.sun.wavelength_m
surfaces = [ocean_surface(wave), dusty_ground(wave)]
```

Prepare both cases together on CUDA. A batch shares its wavelength grid
and layer count. Warp supports 2–16 angular ordinates per hemisphere
and the NumPy solver supports 2–32. Paired differences also require matching
interface heights, spectral weights and band edges. Set case 0 (ocean)
as the reference for both. `enqueue()` solves, integrates over wavelength,
and subtracts the requested reference on the device. Check the result and
read the top-interface flux change
for case 1 (soil). Component 2 is the flux reflected upward.

```python
from particula.optics.gpu.shortwave import prepare_shortwave

columns = [earth.column, earth.column]
batch = prepare_shortwave(
    columns, earth.sun, surfaces, device="cuda:0", reference_cases=[0, 0]
)
batch.enqueue()
batch.check()
top = batch.broadband_delta[1, :1].numpy()[0]
print(f"{top[2]:+.1f} W/m² at the column top")
```

The dusty ground returns 258.2 W/m² more sunlight to the top of the
clear column than the ocean does. To use the result in another device
kernel, pass its CUDA stream as `stream=` during preparation. Subsequent
work on that stream can read `batch.broadband` and `batch.broadband_delta`
directly. Both have axes (case, interface) and components direct downward,
diffuse downward and upward in W/m². Use `batch.download()` for the full
spectral results on the host. For a satellite band, also pass
`view_zenith_deg`, `relative_azimuth_deg` and a `sensor_response` on the
wavelength grid. The device array `batch.band_radiance` then gives one
response-weighted radiance per case in W m⁻² sr⁻¹ m⁻¹.

To follow changing particles, bind a diagnostic to a prepared simulation.
The following uses the one-box aerosol and condensation settings from the
[optical history example](../../studies/validation/material-properties/optical-recording/dense_optical_history.py).
Run it from a checkout after the setup above. Both cases treat the live
particles in box 0 as uniform throughout the lowest kilometre. The upper
layer keeps its reference molecular optics. A `-1` in the top-first box
map skips particle contributions to that layer.

```python
from runpy import run_path

from particula.optics import OpticalConfig
from particula.optics.gpu.shortwave_resident import ResidentShortwave

setup = run_path(
    "docs/studies/validation/material-properties/optical-recording/"
    "dense_optical_history.py"
)
aerosol, condensation = setup["chamber"]()
config = OpticalConfig(wave, particle_index=1.5 + 0.02j)
with condensation.prepare(aerosol, time_step=0.1, device="cuda:0") as sim:
    diagnostic = ResidentShortwave(
        sim, config, columns, earth.sun, surfaces,
        box_indices=[[-1, 0], [-1, 0]], reference_cases=[0, 0],
        coefficient_capacity=48,
    )
    sim.run(steps=1)
    diagnostic.enqueue()
    diagnostic.check()
    wet_top = diagnostic.workspace.broadband_delta[1, :1].numpy()[0]
    print(f"{wet_top[2]:+.1f} W/m² at the column top")
```

With the absorbing particles present, the contrast between surfaces falls
to 113.4 W/m². Reuse the diagnostic after later simulation steps to
recompute the particle optics and fluxes on the simulation stream. The
[batch timing study](../../studies/validation/spectral-radiation/benchmark.md#how-many-columns-benefit-from-cuda)
compares prepared transfer time across different numbers of columns.

## Device calculations

Run the [CUDA optics example](../../studies/validation/material-properties/cuda-optics/cuda_optics.py) for explicit
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

## Absorption and a particle energy balance

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

## Record an optical history during a simulation

Use an optical observer with a CPU simulation or a fixed-step Warp owner.
The observer evaluates the live particle population on its owning backend.
The recording schedule and the snapshot schedule are independent. For example,
with a prepared owner using a 0.1 s timestep, the following captures three
wavelengths at 10 Hz, including the initial state.

```python
from particula.execution.optical_recording import (
    OpticalHistory,
    OpticalObserver,
    read_optical_history,
)
from particula.execution.prepared_recording import run_prepared
from particula.optics import OpticalConfig

config = OpticalConfig([450e-9, 550e-9, 700e-9], particle_index=1.5 + 0.02j)
history = OpticalHistory("optics.sqlite", config, variables=("extinction",))
observer = OpticalObserver(owner, history, batch_size=32)
run_prepared(owner, 10.0, optics=observer, optical_initial=True)

for metadata, arrays in read_optical_history(history.path):
    time_s = arrays["time"]
    extinction_per_m = arrays["extinction"]  # time, box, wavelength
```

Use `run_until` with the same optical arguments for a `CPUSimulation`.
Supply `optical_output_times` for a sparse optical grid. A prepared owner
requires those times to lie on its existing step grid. Explicit CPU output
times may shorten cycles to reach those times. The dense schedule simply
observes each completed step. Manual calls to `observer.record()` can
select additional accepted states. Call `observer.flush()` before closing
the owner when recording manually.

The history commits complete batches and stores wavelengths, units, index
configuration and caller-provided provenance. On CUDA, the observer transfers
bulk optical rows and status in batches and reads no full-state snapshots.
Set `coefficient_capacity` to a bound appropriate to the simulated size
range to control Mie workspace memory. An insufficient bound raises an error
when the batch is checked.

Use `species_index` with axes (wavelength, material) to recompute index
from current composition. Supply indices in the order of the particle
material axis, including the water index for particles that take up water.
The homogeneous mixture uses volume fractions of each material and the
Lorentz–Lorenz function of its dielectric constant.

## Fit bulk optical measurements

Start from a size spectrum and optical observations from a compatible sampling
window. Normalize gas-volume conditions, align instrument RH and wavelengths,
and choose an uncertainty model. An `OpticalGrid` takes number density
integrated over each bin in m⁻³. Use `OpticalGrid.from_diameter_bins` to
convert a spectrum reported as dN/dlog₁₀D using its diameter edges. For
weighted particles, use the [native spectrum functions](../data-model/index.md)
to calculate integrated densities on those edges first.

`EffectiveIndexModel(grid)` fits a common complex refractive index using
sizes measured at the optical RH. To model dry material, supply the optical
`relative_humidity` and either a prescribed `kappa` or bounds for fitting
it. A `sizing_relative_humidity` accounts for water retained in the sizing
instrument under the assumed equilibrium model. A prescribed diameter
`growth_factor` can also supply the radius and water-volume change.
The fitted index is shared across the specified wavelengths.

`MixingStateModel` fits one internal-mixing number fraction, or two fractions
split by one dry diameter. Supply dry species volume fractions, indices and
hygroscopicities. `dry_volume_fractions` converts a mass composition using
species densities. This model redistributes material among equally sized dry
spheres at each bin. Each candidate has the same total dry count and species
inventory. Wet growth is evaluated separately for each subpopulation.
When the data only constrain the bulk composition, the allocation of
composition across dry sizes is an explicit model input.

Use `OpticalObservations` to form the measurement channels from the bulk
extinction, scattering and absorption coefficients. It supports wavelength
or time averaging and a full measurement covariance. Use a truncation-corrected
total scattering observation for comparison to the modeled total scattering.
An angular-channel response requires the differential scattering calculation
above. Treat derived and independently measured coefficients according to
their covariance when forming the residual vector.

Use `fit_optical` with explicit bounds and multiple starts. Select
`device="cuda:0"` on the model for the Warp adjoint. It supports
float64 gradients through the optical and humidity calculations. Use the
local sensitivity, `profile_optical`, and held-out measurements to check which
parameters can be distinguished. Profile intervals depend on the fitted
population, searched bounds and assumed errors. A smooth closed-condensation
trajectory can use `condensation_optics` with an optical time-series adjoint.

The [optical integration studies](../../studies/validation/material-properties/optical-integration.md)
show a measured multi-RH fit, a case where bulk optics leave a size-band mixing
fraction unresolved, and recovery of the initial radius from a time series.

## Feed light absorption back into evaporation

Pass `water_vapor="water"` to `OpticalHistory` when the named native gas
species is water. The reader returns `relative_humidity` on (time, box), as
the ratio of gas-water partial pressure to Buck saturation pressure at
the live gas temperature. On the prepared backend, the optional RH values
are transferred with the other reduced observations. Values above one
represent supersaturation.

Use `OpticalHeating` from `particula.dynamics.optical_heating` to compute
the effect of prescribed illumination on a particle-resolved population of
warm solution spheres. Each resolved slot stores its temperature and
absorbed power. Configure two particle materials (water and a nonvolatile
absorbing solute), the water gas name and a band-integrated irradiance
[W m⁻²] for each wavelength in the `OpticalConfig`. Call
`heating.attach_fields(aerosol)` once before
starting a new simulation. For a resident run, register `OpticalHeating` and
its `OpticalHeatingAdapter` from
`particula.dynamics.optical_heating_gpu` on a `ProcessRegistry` and pass it
to `prepare`. The staged resident operation supports exact checkpoints.

The runnable evaluates absorption at the start of its position in the
sequence. It solves a per-particle temperature balance with gas conduction,
grey infrared emission, sensible heat and latent cooling. The removed
water enters the mapped gas lane using the native number measure.
Temperature [K], absorbed power [W per physical particle] and cumulative
energy transfers [J m⁻³] are available through the registered fields. The
optical observer reevaluates final geometry if recording after the accepted
step. The [illuminated evaporation example](../../studies/validation/material-properties/optical-heating/illuminated_evaporation.py)
checks the two timings and both inventories separately.

This process assumes a prescribed optically thin source, continuum transport,
a liquid water/solute surface activity and one-way evaporation. Geometry
and gas-water concentration are frozen during each temperature solve.
The gas and longwave radiation temperatures are prescribed reservoirs.
Use a timestep refinement check if evaporation changes sizes or humidity
substantially over one step.
