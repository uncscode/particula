# Reproduce the airborne-residue and probe checks

These CPU analyses reproduce the 100 nm assay and the droplet-sampling comparisons in [the uncertainty study](../../../docs/studies/literature/bai-limonene/uncertainty.md). Run the commands from the repository root using the study environment. Source data, cloud columns and calculated tables go in disposable directories. The study’s main replay and its input preparation requirements are documented [one level up](../README.md).

## Can a half-evaporated particle still react?

Prepare the input directory once. The assay reads the same Wilson volatility classes, reconstructed Bai bulk rates and diameter-dependent internal-light correction at 330 nm as the cloud study.

```bash
python studies/bai-limonene/replay.py /tmp/bai-study inputs
python studies/bai-limonene/checks/residue.py /tmp/bai-study /tmp/bai-checks
```

The first test follows 100 nm particles at 300 cm⁻³ through dark evaporation. A gas lifetime of 15 s or 24 h brackets the gas-side boundary. At 50% initial dry mass the script reports the photon-corrected action per unit remaining organic mass relative to fresh material. The ranked, independent and inverse assignments give 0.394–0.431, 1.000 and 1.566–1.592 in `residual-action.csv`. Wilson’s humid suspended-particle distribution and Bai’s wavelength-dependent film kinetics provide the starting constraints. [Wilson et al., Table S1](https://doi.org/10.1021/es505331d.s001) · [Bai et al., Tables S8–S13](https://doi.org/10.1021/acs.est.5c16671.s001)

The second test prepares an initial 2 µg/m³ sample under 60× reference lamp intensity to 50% remaining dry mass. The matched receivers start with the same labeled first-generation low-volatility product. Bulk gas present at transfer goes into a separate mass ledger. At the original particle number concentration and a 24 h gas-renewal time, the split sample receives 48 minutes of darkness or 30× illumination. With reactivity set to zero, one quarter, or the full slow-parent rate, the fractions of the marker converted chemically are 0, 0.02430 and 0.09352. Intact molecules in gas and exhaust together account for 0.11835 of the starting marker in the dark receiver. `product-turnover.csv` reports both readouts. The 1× value is a calibrated monochromatic flux giving the fast material its reconstructed standard solar reaction rate. Both simulations check conservation of organic material and of the marker.

## Would either probe resolve an imposed diameter shift?

```bash
python studies/bai-limonene/checks/probes.py /tmp/bai-pace-inputs /tmp/bai-checks
```

The script fetches the FSSP and CAS 2015 CSV and NetCDF files, verifies each archive digest, and pairs independently screened liquid-cloud minutes. It combines the precise CSV time index with the NetCDF size bounds. The longest continuous qualifying event spans 99 minutes on 25 October. For the FSSP, the integral of the reported spectral ordinate times log₁₀ bin width, scaled by 60, reproduces the archived concentration. For the CAS, the observed bin proportions are normalized to the separately reported total at each minute. A shared 3.5–40 µm window gives medians of 8.41 and 5.74 µm. [PaCE 2015 source archive](https://fmi.b2share.csc.fi/records/va6sm-a6894) · [Doulgeris et al.](https://essd.copernicus.org/articles/14/637/2022/)

`pallas-event.csv` reports pooled medians and mean counts by instrument and size window. `pallas-blocks.csv` pairs adjacent five-minute blocks and injects each shift into either member in turn. Spectra are translated with a uniform number per log diameter within each bin, then rebinned and truncated to the instrument window. A positive change in both treatment orders occurred in 9/9 pairs for +1.2 µm on each instrument and in 6/9 FSSP and 8/9 CAS pairs for +0.3 µm.

## Pass the same cloud through both instruments

After the `inputs` stage, run either the single-hour smoke comparison or the July cloud calculation. The first pair of commands checks the observation operator independently at week-hour 12. The second averages the 57 July cloud snapshots over all 168 sampled hours using the midpoint boundaries between snapshots.

```bash
python studies/bai-limonene/replay.py /tmp/bai-study smoke
python studies/bai-limonene/checks/synthetic_probe.py /tmp/bai-study /tmp/bai-pace-inputs /tmp/bai-checks --smoke

python studies/bai-limonene/replay.py /tmp/bai-study cloud
python studies/bai-limonene/checks/synthetic_probe.py /tmp/bai-study /tmp/bai-pace-inputs /tmp/bai-checks
```

The solver output supplies physical wet diameters and droplet number at each height. The observation operator takes either the top-bin spectrum or a trapezoidal average of the spectra over the full 100 m depth. It computes the median after combining the counts. In `synthetic-probes-resolved.csv`, the slow, clean July cloud’s top and whole-depth diameter changes are +1.222 and +0.943 µm with FSSP edges, and +0.0001 and +0.924 µm with CAS edges. At week-hour 12 alone, the four changes are +1.035, +0.801, 0.000 and +0.768 µm. The field analysis measures variability and instrument response in the observed event. The synthetic analysis bins the simulated paired cloud columns using the same instrument edges.
