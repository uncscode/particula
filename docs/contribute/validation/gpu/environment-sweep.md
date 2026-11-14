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

The initial humidity pilot used 16 boxes per condition and seed 412. Halving
dt from 1 to 0.5 microseconds over 3 ms changed water inventories by at most
0.27014% and final vapor by 0.000225%. These passed 1%/0.1% criteria. Maximum
positive/negative attachment differences were 5.8594%/7.8125% of the initial
per-sign ion count; recombination differed by 2.3438%. The predeclared 2%
population criterion therefore failed. This small stochastic pilot is
inconclusive, not a passing timestep validation.

A separately frozen confirmation uses 256 boxes per condition and seed 413,
with the same timestep, horizon and tolerances. The population comparison
includes Monte Carlo variability; it is a sensitivity check, not a confidence
interval or proof of asymptotic convergence. Reproduce the fixed checks with:

```bash
.venv/bin/python -m scripts.gpu_studies.qualify_environment dry /tmp/dry.json
.venv/bin/python -m scripts.gpu_studies.qualify_environment humidity /tmp/humid.json
```

Confirmation and full publication results will be recorded after completion.

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
