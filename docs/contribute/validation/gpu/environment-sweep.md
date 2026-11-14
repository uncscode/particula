# Environmental dynamics: dry aerosols and humid air with ions

The [dry study](../../../examples/gpu/prepared/gpu_dry_environment_sweep.py)
and [humidity study](../../../examples/gpu/prepared/gpu_environment_sweep.py)
are separate physical experiments. Each published example retains only its
current successful output generation; superseded output remains in Git history.
Failed refreshes preserve the previous successful generation.

## Physical workloads

The dry study starts with 2,490,368 unit-count aerosols: 19 conditions × 1024
independent boxes × 128 particles. Sixteen geometric altitudes span 0–86 km;
three additional chambers use 2, 5 and 10 atm at 288.15 K. The ground-level
case supplies the 1 atm chamber control. A hydrostatic US-standard-atmosphere
approximation supplies altitude T/P, including molecular-scale temperature at
86 km. Number density stays fixed at 1e11 m^-3 to isolate transport effects;
this is not an atmospheric abundance profile. The run advances 1000 s in
0.5 s steps, with no water or ions. Bars are ordered by altitude and pressure.

The humidity study starts with 2,621,440 resolved particles in 16,384 boxes:
128 aerosols, 16 positive ions and 16 negative ions per box. Each box reserves
32 additional slots. T is 278.15/298.15 K, P is 70000/101325 Pa, and initial
RH is 40/60/80/90%. Aerosol number density is 5e13 m^-3; each ion sign starts
at 6.25e12 m^-3. A paired source supplies 1e14 pairs m^-3 s^-1. This is an
illustrative dense laboratory-style population, not ambient ion abundance.
The run advances 3 ms in 1 microsecond steps. Water vapor is finite and
unreplenished; prescribed T/P do not respond to latent heat. Vapor diffusivity
is explicitly held at 2e-5 m²/s because the prepared condensation interface
accepts species constants rather than per-box transport laws.

Both studies use matched lognormal dry radius samples (35 nm geometric mean,
1.6 geometric standard deviation, clipped 15–150 nm), density 1400 kg/m³ and
initial aerosol charges -1,0,0,+1. The humidity model adds κ=0.3 water activity,
Kelvin curvature and Buck vapor pressure. Condensation, charged aerosol
coagulation and ion attachment are active together.

## Ion model and normalization

The opt-in `mobility` attachment law uses ion friction e/Z and
Z=Z_ref sqrt(T/298.15) 101325/P, assuming constant collision cross section and
fixed ion composition. Reference mobilities are illustrative 1.4/1.9e-4
m²/(V s), masses 150/90 u and contact radius 0.25 nm. It omits image forces,
three-body trapping, hydration and composition changes. Opposite-ion
recombination remains an explicitly fixed 1.7e-12 m³/s. LF2013 defaults and its
reference-condition restriction remain separate and unchanged.

Physical-limit tests exposed an inherited inversion of the Coulomb enhancement
ratio in the shared CPU/GPU diffusive Knudsen number. The corrected definition
is sqrt(kBT*m)/(f*a) * Gc/Gk. With the dimensional fit, it approaches
4πa(kBT/f)Gc in the continuum and πa² sqrt(8kBT/(πm))Gk in the ballistic limit.
The old CPU/GPU parity checks could not detect a shared formula error.
Independent charged-limit tests now cover attraction, neutrality and repulsion;
tiny-Knudsen comparisons use zero absolute tolerance.

The changed rates also exposed a CPU sampler defect: a mixed radius bin could
use an ion as both donor and receiver and join two repelling aerosols through
a spurious chain. Actual pair radii now determine donor orientation before
uniqueness, and self-pairs are excluded. The existing no-spurious-calcite
regression is retained. A prior integration expectation of such chains was
corrected to require zero calcite-calcite events in that fixed-seed case.

## Numerical qualification

Hardware: NVIDIA GeForce RTX 5060, 8 GiB; Warp 1.11.1 and Python 3.12.12.
CUDA runs use the existing environment and
`WARP_CACHE_PATH=/tmp/particula-charged-warp`. Restricted sandbox CUDA skips
are not counted as hardware evidence.

Dry refinement uses 64 independent boxes per condition, seed 513, 1000 s,
and dt=0.5/0.25 s. Every individual paired-box approximate 95% interval for
the number-loss difference must lie within ±2 percentage points of initial
number. All 19 conditions passed after the normalization correction.
Intervals are individual, not a simultaneous 95% guarantee.
The largest absolute interval endpoint was 1.4491 percentage points.
`docs/data/gpu/environment-dry-refinement.json` retains all nineteen comparisons.

The initial humidity pilot used 16 boxes per condition and seed 412. Halving
dt from 1 to 0.5 microseconds over 3 ms changed water inventories by at most
0.27014% and final vapor by 0.000225%. These passed 1%/0.1% criteria. Maximum
positive/negative attachment differences were 5.8594%/7.8125% of the initial
per-sign ion count; recombination differed by 2.3438%. The predeclared 2%
population criterion therefore failed. This small stochastic pilot is
inconclusive, not a passing timestep validation.

A separately frozen confirmation uses 256 boxes per condition and seed 413,
with the same timestep, horizon and tolerances. It passed: water differed by
at most 0.26361%, vapor by 0.0003033%, and aerosol number by 0.01526% of initial
aerosol count. Positive attachment, negative attachment and recombination
differed by 0.95215%, 1.90430% and 0.80566% of initial per-sign ion count,
respectively. The negative-attachment comparison is close to the 2% criterion;
passing it does not establish precision substantially below that threshold.
The records are retained in `docs/data/gpu/environment-humidity-pilot.json`
and `docs/data/gpu/environment-humidity-refinement.json`.
The population comparison
includes Monte Carlo variability; it is a sensitivity check, not a confidence
interval or proof of asymptotic convergence. Reproduce the fixed checks with:

```bash
.venv/bin/python -m scripts.gpu_studies.qualify_environment dry /tmp/dry.json
.venv/bin/python -m scripts.gpu_studies.qualify_environment humidity /tmp/humid.json
```

## Completed humidity publication

The full CUDA run completed with seed 411 and 3 ms of physical evolution.
Generation `run-20261114T205354713341Z` records revision `5011becc0` and
source/input hashes. All eight figures were inspected. Starting with 2,097,152
aerosols plus 524,288 ions, it recorded 334 aerosol mergers, leaving 2,096,818
aerosols (0.01593% number loss). The maximum per-box species-mass residual was
8.593e-14 and total charge was conserved exactly at every observation.

There were 46,354 positive and 58,889 negative attachments, 6,861 recombination
events and 12,625 injected ion pairs. Attachment therefore changed aerosol
charge much more often than aerosol mergers changed particle number. The
attachment balance transfers net -12,535 elementary charges to aerosols,
compensated by the remaining free ions. Faster negative-ion depletion is
consistent with the prescribed larger negative mobility; this does not isolate
mobility from the other coupled model processes.

The pooled median diameter rose from 74.453 to 85.209 nm. Water uptake drives
this bulk size change: only 0.01593% of aerosols were lost to mergers. Final
median diameters at initial RH 90% were about 101.84 nm at 278.15 K and
103.19 nm at 298.15 K, compared with about 74.26 nm at RH 40%. The finite
vapor reservoir leaves final RH near 89.28% and 89.77%, respectively, in the
90% cases. Size distributions approach a plateau while attachment and aerosol
mergers continue. Charge-change panels reveal redistribution that is difficult
to see in the nearly flat pooled neutral/singly charged fractions.

Each condition recorded only 12–31 aerosol mergers. The bars follow physical
T/P/RH order; their noisy ranking does not establish significant environmental
differences or an ion-induced coagulation enhancement. That causal comparison
would require matched no-ion controls and uncertainty estimates.

The refresh took 7450.46 s, including 7431.77 s synchronized stepping and
2.05 s preparation. It overlapped other CUDA work, so this is provenance,
not a standalone performance benchmark. Saved source, declared helper inputs
and figure hashes match; only the final successful generation is retained.
The extracted download independently passed a real-CUDA smoke run with
10,240 particles over 100 microseconds and all eight plotting paths. That
smoke exercised both attachment polarities, recombination and the ion source,
with exact charge conservation and a 2.442e-15 maximum mass residual.

## Completed dry publication

The full dry CUDA run completed with seed 512, 2,490,368 initial aerosols and
1000 s of physical evolution. Its saved generation is
`run-20261114T180838997295Z`, with revision `8aa63a3fb` and executable/input
hashes in the manifest. All six figures were inspected. It recorded 449,485
mergers, leaving 2,040,883 aerosols: an 18.04894% total number loss. The largest
per-box material residual was 1.554e-15; every box conserved net charge exactly.
The pooled number-median diameter increased from 70.121 to 75.853 nm.

The altitude response was nonmonotonic. Number loss rose from 9.875% at ground
level to 24.392% at 50 km, then fell to 21.464% at 86 km. This is consistent
with the model approaching its low-pressure collision limit while the reference
temperature decreases above the stratopause. It is not evidence that ambient
mesospheric aerosol concentrations or lifetimes follow these values: initial
number density was deliberately held constant and omitted processes remain
absent. The separate isothermal chamber series decreased monotonically from
9.875% loss at 1 atm to 6.647%, 4.482% and 3.853% at 2, 5 and 10 atm.

The time histories add information beyond those endpoints. The pooled collision
rate declines as the particle population is depleted, while the size quantiles
increase. Absolute neutral and singly charged counts decline; small populations
with charge magnitude at least two emerge. A nearly unchanged neutral fraction
therefore does not imply an unchanged neutral population. These are descriptive
fixed-seed observations, not significance tests or a neutral-control comparison.

The complete refresh took 5357.70 s, including 5345.54 s of synchronized
stepping and 0.92 s preparation. Other CUDA studies ran concurrently, so these
times are execution provenance rather than standalone throughput measurements.
Maintained, saved and downloadable Python sources match; ZIP helper bytes match
the declared inputs. The extracted ZIP passed a separate CUDA smoke run with
4864 aerosols over 5 s, exercising all six plotting paths outside the checkout.

## Implementation checks

The implementation at `54ad1a812` passed the host CPU/CUDA package suite:
10,093 passed, 13 skipped, 57 deselected and one expected failure, with warnings
treated as errors and 91.60% coverage. The skipped and deselected cases are
not claimed as executed CUDA evidence. Independent charged continuum and
ballistic limits supplement CPU/GPU parity; LF2013 default behavior is retained.

The examples/tooling run passed 516 tests with one skip. Its isolated wheel
installation test initially failed because the validation wrapper exported the
checkout through `PYTHONPATH`. Rerunning the seven bundle tests without that
override passed, including the wheel and offline source-distribution checks.
Ruff lint/format, mypy (589 files) and documentation structure checks passed.
The final website build rendered 459 pages and verified 589 HTML pages,
272 API modules, 1860 legacy anchors and 130 redirects with zero errors.
All three study downloads match their maintained Python sources and declared
helpers; separately extracted bundles passed the CUDA smoke checks described
here and in the charged-coagulation validation record.

## Diagnostics and interpretation boundaries

Each observation checks every box's species inventories, including water gas,
injected ion material and neutral-product exports, at 2e-11 relative tolerance.
Total aerosol-plus-ion charge is exact. Aerosol charge alone can change.
Aerosol spectra exclude free ions and neutral products. Histograms use
`dN/(N_initial dlog10 D)` so their integral shows number loss. Charge counts
and histogram counts reconcile independently with the represented population.

Aerosol N0−N counts aerosol-aerosol mergers. Free-ion balances separately count
positive/negative attachment and recombination, checked against ion material
attached to aerosols. Recombination creates an inert neutral particle; periodic
cleanup exports its mass. Interval collision rates are averages, not exact
event times. Pooled diameter quantiles and per-condition size/charge histories
show dynamics beyond the final bars. Printed interpretations use actual run
values; rankings do not establish statistically significant environmental
responses, atmospheric validation or CPU/GPU speedup.
