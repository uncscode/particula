# Optics in simulation runs and inverse modeling

Branch `codex/optics-runnable-integration` starts at `51e25ee110`.

## Deliverables

1. Sample live bulk spectra at accepted steps with an independent snapshot
   schedule. Support prescribed and species-volume Lorentz–Lorenz indices,
   configurable output wavelengths and a bounded chunked history on CPU and
   resident CUDA. Qualify a 10 Hz condensation example with sparse offline
   spectra, output sizes, host/device memory and runtime. Include optical RH
   provenance.
2. Couple spectrally integrated particle absorption to optically thin heating
   and water evaporation. Qualify a nonvolatile absorbing solute with water,
   fixed gas temperature, gas heat exchange, thermal emission and latent
   cooling. Test light-on/off, conservation and timestep refinement on CPU
   and CUDA. Record absorption and state at the correct process time.
3. Fit a homogeneous effective complex index at one wavelength with paired
   size and optical measurements. Add joint dry index and effective kappa
   with a prescribed or fitted subsaturated growth factor. Compare a
   well-constrained set of independent channels to extinction alone, and
   withhold an RH or wavelength. Include size and number calibration
   sensitivity.
4. Fit internal/external mixing fractions with prescribed component indices,
   densities, hygroscopicities and dry species masses. Preserve total number
   per dry bin and species inventory at both endpoints. Fit one common
   fraction, then two dry size-band fractions. Compare held-out predictions,
   uncertainty and alternative size allocations. Demonstrate a weakly
   identifiable case with near-identical component indices.
5. Use mixed extinction/scattering/absorption losses on CUDA with a bounded
   Warp tape. Validate local and complete loss derivatives, including a
   checkpointed fixed-step condensation trajectory whose initial size is
   optimized against bulk optics. Compare synchronized time, forward/backward
   evaluation counts and memory against finite differences. Reserve physical
   radius and gas histories for validation.
6. Publish five maintained percent-format examples for dense chamber optics,
   illuminated evaporation, effective index and growth, mixing state, and
   trajectory inversion. Save compact figures and results and connect examples
   to the topic graph. Run targeted and broader validation, self review the
   complete change, and open a pull request.

## Suchdol version 2 data audit

The [public file listing](https://data.mendeley.com/datasets/8ds98t2f3x/2)
is CC BY 4.0. Five downloaded source files matched their listed SHA-256 sums.

- The combined size and scattering-enhancement file has 6,671 distinct
  interval centers and 71 size columns labeled `bin_01` through `bin_71`.
  The combined dry/wet optical file has 6,670 centers. There are 6,363
  matching labels with all size values present before optical validity checks.
  Use interval bounds as well as centers when matching. The public column
  names do not encode the diameter grid or population measure.
- The separately averaged dry and wet files contain 4,618 and 4,617 rows,
  respectively, with optical-cell RH and temperature. The instrument
  alternated between dry and humidified samples. Match explicit averaging
  windows and check coverage and population stability before treating a pair
  as two observations of one population.
- The [instrument paper](https://amt.copernicus.org/articles/19/1611/2026/)
  reports five-minute mobility spectra over about 10–800 nm, at 32
  channels per decade. Verify the exact size columns, measure and sizing RH
  from source metadata before fitting. Its total scattering coefficients
  were truncation-corrected and standardized to STP. Confirm the concentration
  basis for the paired size spectra. Keep any modeled extinction derived
  from scattering and absorption out of the independent residual vector.
  Use EC/OC as partial composition evidence.

## Verified foundation and remaining work

The branch implements dense or scheduled live-state bulk histories and
batched device transfer on the prepared owner. Optical recording tests compare
resolved counts against independent Number/volume weighting, exercise durable
continuation, prohibit full-state snapshot calls, and compare canonical
prepared checkpoints for passive dense sampling. A condensation test compares
three saved optical wavelengths against sparse CPU calculations from the
corresponding device snapshots.

The tiled Warp bulk adjoint evaluates existing Mie coefficients and derives
local slopes from the Riccati–Bessel equation. Its tape hook recomputes each
tile in reverse and propagates gradients for radii, complex indices and
number-density weights. Forward comparisons and directional-difference tests
passed on Warp CPU and an RTX 5060 with Warp 1.11.1. The broader optics and
recording validation returned 318 passes with CUDA available. Two additional
CPU humidity-equilibrium checks passed against independent scalar roots and
the flat-interface limit.

Next wire the bulk adjoint into the observation and fitting APIs, including
growth and composition derivatives. The size-bin audit, physical feedback,
checkpointed optical trajectory, published examples, full validation,
self review and PR are still open.

## Retrievals, optical trajectory, and measured inputs

The parameters now run through a Warp tape and the tiled Mie adjoint. The
effective-index fit handles measured-at-RH radii, prescribed growth
factors, specified or fitted κ, and nonzero sizing RH. The mixing-state fit
preserves number and dry species inventories for arbitrary prescribed
volume fractions per bin. Both use observation covariance, project
spectral/time channels, and report bounded least-squares sensitivity and
profile intervals. The checkpointed inverse retains only the selected optical
observations and the mass checkpoints needed to replay gradients.

EBAS station CZ0004B provides the 71 instrument diameter centers (8.8 to
835.7 nm), hourly size spectra and measurement RH for the campaign years.
Its metadata states STP, Nafion drying below 40% RH, and 20% intercomparison
uncertainty. The standard sizing protocol reports number density per decade
of mobility diameter. The published campaign spectra agreed best with
three-hour means of the extracted EBAS spectra at zero time offset over the
four audited days. The example uses individual hourly size distributions
and corrected nephelometer total scattering for two dry–wet–dry windows.
It fits 450/550 nm channels and campaign-processed dry 550 nm absorption,
then predicts held-out 700 nm scattering. Bin edges are approximated from
the published centers. PM₁₀ optics include size contributions above the
measured mobility range. The small reproducible fixture carries archive
locations, slice offsets, and original source file hashes.

- [Hourly EBAS archive](https://doi.org/10.48597/GHVT-PE5Z)
- [Instrument methods and field limitations](https://amt.copernicus.org/articles/19/1611/2026/)
- [Optical studies and worked examples](../docs/studies/validation/material-properties/optical-integration.md)

Remaining scope includes the active optical heating and evaporation process
and worked example, full repository validation, integrated self-review, and PR.
