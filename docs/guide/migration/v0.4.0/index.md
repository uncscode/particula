---
title: Migrate particle populations from 0.3.6 to 0.4.0
---

# Migrate particle populations from 0.3.6 to 0.4.0

In 0.4.0, each box stores the particles represented in its air sample. When
upgrading a study, identify which inputs are particle counts and which are
measurements per unit air volume. The [input and writer example](../../data-model/index.md#migrating-particle-inputs-and-writers-from-036)
shows how to import a measurement, edit raw counts and replace a sample. Then
compare your original number density and species mass budget at two different
sample volumes. A test confined to 1 m³ can hide a missing volume conversion.

## Choose the quantity your calculation needs

Let `p` be a CPU `ParticleData`, with B boxes, N bins, resolved slots or
radius nodes, and S materials. Each `p.volume[b]` is the positive represented
air volume of one box in m³. A particle's material volume is
`sum(mass / material_density)` over its materials. A chamber or grid cell has
its own physical air volume, which may differ from the represented sample.

| 0.3.6 input or operation | 0.4.0 expression | Meaning, units and shape |
| --- | --- | --- |
| Construct with raw resolved counts | `ParticleData(..., population=counts)` or builder `set_population(counts)` | Represented number per slot, (B,N). Multiplicities can be fractional. |
| Construct from old discrete bin densities or radius-PDF ordinates | `population=density * volume[:, None]` for already normalized arrays | Number per bin, or represented number per metre of radius (m⁻¹), (B,N). Multiply each row by its represented air volume once. |
| Write raw amounts through `p.concentration[:]` | `p.population[:] = amounts` | Writes the existing raw buffer. For an increment expressed per m³ of air, multiply the increment by the row's volume first. |
| Initialize from a measurement | Builder `set_number_density(values, units)` or a [unit-bearing factory](../../data-model/construction.md#distribution-and-volume-rules) | Bin or slot density in m⁻³, or radius density in m⁻⁴. |
| Replace a measured CPU sample | `p.set_slot_concentration_density(values, volume=new_volume)` | SI density (B,N), or (N,) for one box. Optional scalar or (B,) volume in m³. Omitting volume uses the current volume. |
| Read density at each slot | `p.slot_concentration_density` | Detached (B,N), equal to population divided by row volume. Discrete/resolved values are m⁻³. PDF values are m⁻⁴ and need radius quadrature to contribute to a total. |
| Read the total number density | `p.number_density` | (B,), m⁻³, summed or integrated by representation. Multiply by `p.volume` for represented number per box. |
| Read mass by material | `p.species_mass_density`, `p.species_mass_inventory` | (B,S), kg/m³ air and kg per box respectively. Each slot's `p.masses` is kg per particle and material. Gas remains `gas.concentration` in kg/m³ air. |

Throughout 0.4.x, the second positional argument of `ParticleData`, its
`concentration=` keyword and writable `.concentration` property keep their
legacy meaning. They accept resolved counts at any represented volume, and
binned/PDF raw values only when every box has volume exactly 1 m³. A single
non-unit binned row rejects that legacy access for the entire batch. Keep a
reference to `p.population` to see later in-place writes. A density property
returns a detached result each time you read it.

## Import and replace a measured sample

A discrete bin measured at 2 cm⁻³ contains 500,000 represented particles in
0.25 m³. Import its SI equivalent, 2 × 10⁶ m⁻³, with the base installation.
With Pint installed, replace the density call with
`.set_number_density([2.0], "1/cm^3")` to use the instrument's units directly.

```python
import numpy as np
from particula.particles import ParticleDataBuilder

sample = (
    ParticleDataBuilder()
    .set_distribution_type("discrete")
    .set_masses(np.array([[[1e-18]]]))
    .set_density(np.array([1000.0]))
    .set_number_density([2e6], "1/m^3")
    .set_volume(0.25)
    .build()
)
np.testing.assert_equal(sample.population, [[500_000]])
```

A replacement measurement of 400 m⁻³ represents 100 particles in the original
sample and 200 in a new sample twice its size. The setter validates both
inputs before committing either one. Other holders of the raw arrays see the
accepted update.

```python
counts, volumes = sample.population, sample.volume
sample.set_slot_concentration_density([400.0])
np.testing.assert_equal(counts, [[100.0]])
sample.set_slot_concentration_density([400.0], volume=0.5)
np.testing.assert_equal(counts, [[200.0]])

with np.testing.assert_raises(ValueError):
    sample.set_slot_concentration_density([-1.0], volume=1.0)
np.testing.assert_equal(counts, [[200.0]])
np.testing.assert_equal(volumes, [0.5])
```

## Check integration, mass and diameter spectra

For discrete bins or resolved slots, sum the per-slot densities. At
[4, 8] m⁻³ the two slots store [1, 2] particles in 0.25 m³ and [16, 32] in
4 m³. Both boxes give 12 m⁻³. To check each material separately, multiply
each slot's integrated number density by its mass of that material and sum
over slots. Multiplication by the box's air volume gives its mass inventory.

The example radius PDF integrates to 9 m⁻³ in either sample. Integrate
over its radius grid in metres. The endpoints receive half their adjacent
interval and interior nodes receive half the sum of their adjacent intervals.
The nodes [1, 2, 4] nm have widths [0.5, 1.5, 1] × 10⁻⁹ m. Multiplying by
the prescribed ordinates [2, 4, 2] × 10⁹ m⁻⁴ gives contributions of [1, 6, 2]
m⁻³. The two samples therefore represent 2.25 and 36 particles in 0.25 and
4 m³. Apply the same weights to per-particle species masses.
The [three-representation tutorial](../../../studies/tutorials/building-an-aerosol/particles/represented_population.py)
runs this calculation with unequal slot masses in two materials.

To compare with an instrument, first establish its diameter definition. For
spherical volume-equivalent sizes, diameter D = 2r (and dn/dD = ½ dn/dr at
D = 2r). For binned spectra, divide each integrated bin density by
Δlog₁₀D to obtain density per decade. Dividing by a linear diameter width
instead gives density per metre. Use the original diameter edges when
recovering a reported band from per-decade data.

For example, five illustrative diameter classes with prescribed density
weights straddle a two-bin band. Interior
edges go to the bin on their right, and the last bin includes the final upper
edge. The band contains 10 m⁻³. Adding 1 m⁻³ below it and 5 m⁻³ above it
recovers the original 16 m⁻³.

```python
from particula.particles import get_binned_number_spectrum, get_number_per_decade

diameter = np.array([1, 2, 4, 8, 9]) * 1e-6
edges = np.array([2, 4, 8]) * 1e-6
bins = get_binned_number_spectrum(diameter, [1, 2, 3, 5, 5], edges)
per_decade = get_number_per_decade(bins.bin_number_density, edges)
band = np.sum(per_decade * np.diff(np.log10(edges)))
np.testing.assert_allclose(bins.bin_number_density, [2, 8])
np.testing.assert_allclose([band, bins.below_range + band + bins.above_range], [10, 16])
```

## Follow the volume through a process

Flushing and expansion can produce identical final densities while leaving
different inventories. Start with 100 represented particles and 1 µg of gas
in 0.25 m³. At fixed volume, one clean-air half-life removes half of each.
Expansion of the original resolved sample to 0.5 m³ retains all 100 particles
and the full 1 µg. Both endpoints report 200 particles/m³ and 2 µg gas/m³.
Check the represented number and gas kg as well as both densities. The
[dilution guide](../../processes/dilution.md#compare-a-flush-with-physical-expansion)
runs the fixed-volume CPU and prescribed-volume Warp calculations.

Replacing a particle measurement changes the particle counts. Editing only
the particle volume leaves the stored counts and gas density in place.
For the same expansion above, such an edit would imply 2 µg of gas in the
new volume. Numerical representative-volume scaling has a different purpose.
When the resolved slot-exhaustion policy selects it, the sample's counts and
volume scale together, preserving pre-existing particle densities and gas
concentration. Ordinary resampling combines or redistributes slot weights
at the current volume. [Slot-exhaustion policies](../../gpu/slot-exhaustion.md#native-population-boundary)

- For device calculations, transfer discrete or resolved counts together with
  per-box volume. `WarpParticleData.population` holds raw amounts. Import
  measured densities on the CPU before upload. Native continuous radius PDFs
  stay on the CPU, and each device process has its own weight requirements.
  [GPU support](../../gpu/support.md) · [Explicit transfers](../../gpu/transfers.md#particle-transfer-boundary)
- On parcel export, use the volume at the selected time. For grid transport,
  multiply sample density by physical cell volume to obtain cell inventory.
  [Parcel and grid budgets](../../processes/parcels.md)
- For optics, sum the product of each cross section (m²) and its integrated
  number density (m⁻³) to recover the bulk coefficient (m⁻¹).
  [Native particles and measured arrays](../../optics/index.md#population-coefficients-and-path-geometry)

## Reopen your study

Record `particle_population` and `volume` together. Open 0.3.6 analysis files
with `load_analysis`. Its returned metadata identifies the original schema
in `source_analysis_version` and the original units and axes in
`source_observables`. For a derived-only history, compare the recorded total directly.
Continue an exact 0.3.6 checkpoint with its original software, runtime and
supported device. To start a 0.4.0 calculation from that study, construct a
new physical state and retain the old checkpoint as provenance.
[Record and read analysis histories](../../getting-started/recording-restart.md#read-a-036-particle-history)

Before replacing a figure, reconstruct one unrounded point from its old
inputs, then repeat the calculation with the migrated state at the same
time, bin edges and units. Check the tails and each species budget alongside
the observable. The [study comparisons](../../../studies/validation/workflows-and-releases/population-v0.4.0.md)
record the reference data and comparison for individual studies.
