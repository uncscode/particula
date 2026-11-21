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
