# Record and infer aerosol properties from optics

Dense optical histories retain the quantities seen by an optical instrument.
They can follow a changing population, test a physical model against spectra,
and support retrievals with a much smaller output than a particle history.
Use the worked examples to check each of those steps. The numerical
scattering model is Lorenz–Mie theory for homogeneous spheres.

- [Record extinction at 10 Hz](optical-recording/dense_optical_history.py).
  Compare continuous optical recording with sparse offline checks during
  hygroscopic growth.
- [Warm a droplet with light](optical-heating/illuminated_evaporation.py).
  In the prescribed 24 ms pulse, the larger solution drop warms by 1.82 K
  and shrinks from 6.00 to 5.81 µm. Independent pre-step absorption,
  paired-phase water, and energy checks qualify the CPU and CUDA feedback.
- [Retrieve effective index and growth](optical-retrieval/infer_refractive_index.py).
  The synthetic κ interval narrows from 0.15–0.43 with one humidified RH to
  0.26–0.30 with dry and two humidified conditions at 2% prescribed errors.
  The Suchdol example retains 700 nm for a field prediction. The November
  fit overshoots that channel by 12–17%, while the August fit undershoots
  by up to 7%. These results depend on the assumed size calibration,
  homogeneous spheres and one effective index across the visible band.
- [Ask whether the optics resolve mixing state](optical-mixing-state/infer_mixing_state.py).
  Two populations can share dry sizes and each species inventory while
  distributing the components differently among particles. The example
  fits a constrained family and uses measurements on selected sizes to
  test whether both size-band fractions can be resolved.
- [Fit initial radius from the optical trajectory](../condensation/optical-trajectory/infer_optical_trajectory.py).
  The fixed-step closed-box inverse retains physical radius and vapor
  for an independent check and compares CUDA autodiff with finite
  differences of the optical loss.

The [optical guide](../../../guide/optics/index.md) defines the optical
quantities and recording interfaces. The [reference corpus](sphere-optics/index.md)
and [CUDA qualification](cuda-optics/index.md) establish forward accuracy
for their respective numerical domains. Each retrieval example reports
its own sensitivity or profile and held-out prediction.

## Prague-Suchdol optical retrieval inputs

The effective-index example download includes
`docs/data/optical-retrieval/suchdol.json` with two three-hour reporting
windows from the measurement campaign. Both underlying data sources are
CC BY 4.0. The fixture preserves published diameter centers, hourly number
spectra, relative humidities, temperatures, and optical coefficients.
Its metadata carries the versioned source hashes and the exact EBAS
resource name.

- Lenka Suchánková, *Humidified single-nephelometer set-up system datasets*,
  version 2. Hourly scattering, temperature, and optical-cell RH come from
  `final_averaged_data_dry.csv` and `final_averaged_data_wet.csv`. The 550 nm
  absorption estimate comes from `abs550.dry` in `dry_wet.csv`. The source
  tables use semicolon delimiters and decimal commas. The paired optical
  rows are dry, wet, and dry, beginning at the UTC times in each fixture
  window. The 550 nm wet scattering entry also agrees with the corresponding
  combined-table row to the table's precision. [Campaign dataset](https://doi.org/10.17632/8ds98t2f3x.2)
- Jakub Ondráček et al., *Particle number size distribution at Prague-Suchdol*,
  EBAS CZ0004B, level 2. The fixture takes the diameter centers from `D`
  and the hourly standard-volume spectrum from
  `particle_number_size_distribution_amean`. The in-instrument RH comes
  from `relative_humidity`. Match the lower edge of `time_bnds` in UTC
  (days since 1900-01-01). For the two archived ASCII subsets, read all
  `D` bins and the time indices from `ebas_subset_offset-4` through
  `ebas_subset_offset+27`, inclusive, for size, RH, and temperature.
  [EBAS record](https://doi.org/10.48597/GHVT-PE5Z)

The inference script converts the mobility spectrum from cm⁻³ per decade
to bin-integrated m⁻³ using log-midpoint edges. It compares corrected total
scattering on the shared standard-volume basis. See the instrument paper
for optical truncation, sequencing, and the difference between the diameter
coverage of the size spectrometer and the nephelometer inlet. [Instrument paper](https://amt.copernicus.org/articles/19/1611/2026/)
