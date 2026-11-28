# Migrate Particula guides and scientific studies from 0.3.6 to 0.4.0

**Outcome.** A user can migrate a running study, identify which volume belongs in each calculation, and reproduce its published physical comparison. Deliver a discoverable migration guide, a worked three-representation example, and a study-by-study validation record. The core storage change and minimum upgrade example merged in [#1756](https://github.com/uncscode/particula/pull/1756). This plan covers the follow-through in [#1753](https://github.com/uncscode/particula/issues/1753), building on the [consumer inventory](issue-1751-particle-population-consumers.md) and the [storage migration](issue-1752-extensive-particle-population.md).

## Starting point

Use upstream merge `0fa70c4964` for the 0.4.0 implementation and tag `v0.3.6` (`3435f653ae`) for the old library contract. Use each study's own recorded revision and inputs for its scientific baseline. The [data-model guide](../docs/guide/data-model/index.md#migrating-particle-inputs-and-writers-from-036) already shows measured import, direct mutation and density replacement. Link to it immediately from the upgrade guide. The [release workflow](https://github.com/uncscode/particula/actions/runs/46148955644) published [Particula 0.4.0](https://pypi.org/project/particula/0.4.0/) on November 28, 2026. Its wheel contains the minimum input and writer instructions in the data-model guide. The dedicated migration page in this change becomes available on the website after merge and in a subsequent package build.

A first pass over tracked Python files found the following literal `.concentration` references, including gas, comments and measurements. These are search candidates. Dynamic field names, helper calls, constructor keywords, schema keys and plot labels need a separate pass.

| Directory | Python files with matches | Occurrences |
| --- | ---: | ---: |
| `docs/studies/` | 65 | 293 |
| `studies/` | 16 | 48 |
| `scripts/` | 30 | 100 |
| `examples_tests/` | 18 | 89 |

## 1. Publish the upgrade guide and its executable example

Add `docs/guide/migration/v0.4.0/index.md`. Link it from the data-model, construction, ownership and migration entry points and register its navigation and topics. Add one worked example under the particle data tutorials, its registry entry and an example test. Use the following contract for the guide's old-to-new table. Let B denote boxes, N slots or radius nodes, and S particle species.

| Meaning or operation | 0.4.0 choice | Units and shape |
| --- | --- | --- |
| Construct or replace raw population | Constructor `population=`, builder `set_population`, in-place `p.population[:]` | (B,N). Represented number per discrete bin or resolved slot, including fractional multiplicity. For a continuous radius PDF, dN/dr in m⁻¹. |
| Migrate 0.3.6 raw input | Old resolved counts go directly to population. Multiply old discrete densities or PDF ordinates by represented air volume once. Legacy second positional argument, `concentration=` and writable `.concentration` retain legacy semantics in 0.4.x. | Legacy raw access supports resolved populations and unit-volume bins/PDFs. A non-unit binned row rejects legacy raw access for the batch. Document raw-buffer ownership and detached derived reads. |
| Import or replace measured concentration | Builder `set_number_density` or unit-bearing factory input. On an existing CPU container use `set_slot_concentration_density(values, volume=...)`. Omit volume to use its current value. | Per slot m⁻³, or dn/dr in m⁻⁴ for a radius PDF. The instance setter accepts SI. Supply (B,N), or (N,) for one box. A supplied volume is scalar or (B,) m³. |
| Read per-slot or integrated density | `slot_concentration_density` gives P/V. For a PDF, weight each ordinate by its trapezoidal node width in metres. For discrete and resolved slots the density itself is the contribution. | Per-slot read (B,N), m⁻³ or m⁻⁴. Contributions (B,N), m⁻³. Show the widths explicitly. The internal weight helper is private. |
| Read totals and mass | `number_density`, `species_mass_density`, `species_mass_inventory` | (B,) m⁻³, (B,S) kg/m³ air, (B,S) kg per box. Species mass is per particle before population weighting. Gas concentration is kg/m³ air. |

**Acceptance examples.** Run the same cases with represented volumes 0.25 and 4 m³. For discrete or resolved slot densities [4, 8] m⁻³, expect stored [1, 2] and [16, 32], and total density 12 m⁻³ in both boxes. For a radius PDF on [1, 2, 4] nm, use dn/dr = [2, 4, 2] × 10⁹ m⁻⁴. Node widths [0.5, 1.5, 1] nm give 9 m⁻³ and inventories 2.25 and 36. Use unequal slot masses in two species and check the mass budget for each. Show a builder import of 2 cm⁻³ producing 500,000 particles at 0.25 m³, plus its 2 × 10⁶ m⁻³ base-install equivalent. On replacement, 400 m⁻³ stores 100 at 0.25 m³ or 200 with a supplied 0.5 m³. An invalid joint update must preserve both buffers.

Distinguish radius from diameter (D = 2r), differential ordinates from integrated bins, and per-decade density from bin density divided by a linear width. Calculate per-decade density as integrated bin density divided by Δlog₁₀D. Show band and tail closure at bin edges. Keep particle material volume (mass divided by material density), represented sample air volume, and physical chamber or grid-cell volume explicit.

## 2. Explain dilution, transfers and stored data at their call sites

- **Flush and expansion.** Start with 100 particles and 1 µg of gas in 0.25 m³. One clean-air half-life at fixed volume leaves 50 particles and 0.5 µg. Expanding the original resolved state to 0.5 m³ leaves 100 particles and 1 µg. Both report 200 particles/m³ and 2 µg/m³. Show the fixed-volume CPU dilution example and the supported resolved Warp prescribed-volume step. A replacement measurement and a particle-only volume edit have different gas budgets. For numerical resampling, document when represented number and sample volume scale together. [Process guide](../docs/guide/processes/dilution.md) · [Executable comparison](../particula/gpu/kernels/tests/communication_test.py)
- **GPU, coupling and optics.** Document raw `WarpParticleData.population` and per-box volume for discrete and resolved transfers, the CPU-only density setter, and the CPU boundary for radius PDFs. Document which processes admit fractional weights. Trace parcel exports and CFD cell reductions to the actual volume used at each sample. For optics, compare the native population route with an explicit sum of per-particle cross section times integrated number density. Preserve the independent measurement-array API. [GPU support](../docs/guide/gpu/support.md) · [Parcel guide](../docs/guide/processes/parcels.md) · [Optics](../docs/guide/optics/index.md)
- **Analysis and exact restart.** Demonstrate recording `particle_population` together with `volume`, reading 0.3.6 analysis files with `load_analysis`, and checking source-schema metadata and physical units for resolved, discrete, PDF and derived-only fixtures. Explain the exact restart version/schema/device constraints separately. Update the current recorder, restart and Helper references. Keep archived validation pages tied to their original versions. [Recorder guide](../docs/guide/getting-started/recording-restart.md) · [Original fixtures](../particula/execution/tests/data)

## 3. Audit studies and compare physical outputs

Track the audit in a validation page under `docs/studies/validation/workflows-and-releases/`. Inventory maintained producers and consumers, including helpers used by multiple examples. Search raw and derived access, `concentration=`, `counts`, quoted field paths, Warp buffers, old recorder keys, and saved-result readers. Follow each value into its reduction, plot and caption. Give every affected study a row with source and baseline revision, original units, volume owner, migrated expression, reference data, command, seed/device, measured difference and disposition. For intentional legacy reads, measurement data and compatibility tests, state why the existing expression is correct.

| First comparisons to complete | Quantitative acceptance criterion |
| --- | --- |
| Three-representation coagulation tutorial | Replace the tutorial's explicit integration of legacy PDF `.concentration` with `number_density`. Compare each time series against its own baseline and explicit irregular-grid quadrature. Compare resolved particle mass using extensive weights. Preserve the stochastic sample volume, seed and original time grid. [Tutorial](../docs/studies/tutorials/coagulation-populations/coagulation_5_time_evolution.py) |
| CPU dilution | Compare each particle and gas lane with its analytic exponential decay. Add the two-volume comparison above so unit-volume data cannot hide an extra factor. [Example](../docs/studies/examples/particle-processes/sources-and-dilution/cpu_dilution.py) |
| Parcel export and prescribed-flow cavity | At two `aerosol_at` samples, reconcile represented number and species inventory with each sample's volume. Replay the cavity with recorded fields and compare cell-integrated totals and the published observable against its saved arrays. Preserve empty-bin coordinates and distinguish physical cell volume from sample volume. [Parcel checks](../particula/dynamics/tests/parcel_test.py) · [Cavity reference values](../docs/data/prescribed-flow-cavity) |
| Π Chamber and Pallas spectra | Reconstruct Π Chamber's 18 time-zero bins per box from restart cohorts, including support outside the reported edges. For Pallas integrate the 6–13 µm band over its five intervals in cm⁻³ and compare each probe/date with the archived attribution table. Preserve probe response and bin-edge conventions. [Π Chamber checks](../examples_tests/pi_chamber_spectra_test.py) · [Pallas checks](../examples_tests/pallas_number_spectrum_test.py) |
| Jeju chamber | Recover Welas band concentrations from published per-decade spectra over the same valid minutes and optical diameter bins. For run 5 compare the 5–10 µm means against the underlying values (published rounded values 388 measured, 141 pooled, 40.8 individual cm⁻³). Keep instrument events and simulated weights distinct. Investigate any change to the reported deficit before updating the figure. [Published comparison](../docs/studies/literature/jeju-chamber/spectra.md) · [Provenance and replay](../docs/studies/literature/jeju-chamber/replay.md) |

For each deterministic comparison record the maximum absolute difference with units and the relative difference where the reference is nonzero. Use the existing per-study tolerances and unrounded figure data. For stochastic comparisons record the seed and device and use the study's ensemble criterion where appropriate. Repeat affected physics or changed data interpretation on the supported backend. Redraw plotting-only changes from verified numeric data. Retain compact figure values, hashes and retrieval instructions for external measurements. Preserve historical engines and reporting helpers in reproducible downloads. A changed conclusion needs an explanation in the study and the PR before replacing its baseline.

## 4. Verify and hand off

1. Run the three representation examples and the fixed-volume/expansion comparison on CPU and actual CUDA where supported. Then run the owning dilution, parcel, cavity, optics, recorder, and spectrum tests selected by the audit. Run the affected example tests with warnings as errors. Retain device and skip information in the evidence record.
2. Register the guide and worked example. Run `python scripts/check_docs_structure.py`, `python scripts/docs_examples.py check`, `npm --prefix website test` and `npm --prefix website run build`. Inspect the rendered migration table, a radius-PDF figure, a chamber spectrum (axis, units and caption), and an extracted example download.
3. Build the wheel and sdist and run `scripts/check_doc_bundle.py` against both. Verify the migration page is in the installed docs and its example runs from a clean installation. If changing shipped Helper guidance, run its package checks and follow the independent plugin version policy. Use the [repository validation commands](../plugins/particula-helper/references/repository-validation.md).
4. Close [#1753](https://github.com/uncscode/particula/issues/1753) with links to the guide and audit, actual results, numerical differences, and any remaining backend or historical-replay limits. Changes to the storage API, a general-purpose expansion operator or new Warp setters would require separate scope.

**Planning baseline.** At `0fa70c4964` the following focused command passed 40 tests. This establishes the starting point for population and spectra checks. Study replays, rendered-page inspection and release-bundle validation above are implementation work.

```sh
python -m pytest -q --no-cov -Werror \
  particula/particles/tests/extensive_population_test.py \
  examples_tests/pi_chamber_spectra_test.py \
  examples_tests/pallas_number_spectrum_test.py
```
