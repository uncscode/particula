# Two-million-particle environmental sweep

This record accompanies the
[charged aerosol environmental example](../../../examples/gpu/prepared/gpu_environment_sweep.py).
It is a controlled CUDA demonstration, not an atmospheric validation or a
CPU/GPU speedup measurement. It is separate from the explicit-ion Lucky Few
study and does not extrapolate LF2013 attachment rates.

## Model and fixed workload

The source revision is `4a36139cd01aa18fe10b48abad86d7225026c758`.
The default workload contains 2,097,152 initially active resolved particles,
all resident simultaneously, in 16,384 independent well-mixed boxes. Each
condition has 1024 boxes of 128 unit-count particles, at initial number
concentration 5e13 m^-3. Conditions are the Cartesian product of:

- Temperature: 278.15 and 298.15 K.
- Pressure: 70000 and 101325 Pa.
- Initial relative humidity: 0.4, 0.6, 0.8 and 0.9.

The matched initial dry-radius samples have geometric mean 35 nm and geometric
standard deviation 1.6, clipped to 15–150 nm. Dry density is 1400 kg/m3;
water density is 1000 kg/m3. Initial water volume is 0.2 times dry volume.
Charge repeats -1, 0, 0, +1, giving exact zero net charge in every box.

Condensation precedes charged hard-sphere Brownian coagulation. There are
1000 timesteps of 1e-6 s, totaling 0.001 s. Root seed is 411. Water activity
uses dry kappa 0.3, Kelvin curvature with water density, and Buck vapor
pressure. Surface tension is 0.072 N/m, accommodation is one, and vapor
molar mass is 0.018 kg/mol. Vapor diffusivity is a fixed 2e-5 m2/s, an explicit
approximation because prepared configuration accepts species constants rather
than per-box transport laws. Temperature is prescribed and humidity evolves
in a closed water budget. These are illustrative parameters, not measured
material properties or an altitude trajectory.

Helper SHA256:
`bdec86572aa3185c517e17205a176669bedf5a433a5c1b9a51ffa9d1bc8d80a5`.
Percent-source SHA256:
`9390b6c8d061ea2678b191ada927856a07300ac683a65d3999eafbdfa8a32f2f`.

## Numerical qualification

On the NVIDIA GeForce RTX 5060 (8 GiB), Warp 1.11.1, Python 3.12.12:

```bash
WARP_CACHE_PATH=/tmp/particula-charged-warp .venv/bin/pytest \
  scripts/tests/environment_sweep_test.py --no-cov -q -W error
```

Nine tests passed. They cover unit-count geometry and matched conditions,
invalid inputs, CPU/CUDA condensation agreement at 1e-10 relative tolerance,
half-step water-growth sensitivity (1% particle mass and 0.1% gas tolerances),
and a separately written kappa-Kohler equilibrium expression with 0.1%
tolerance. Equilibrium is evaluated using each box's own final humidity,
not its condition's mean humidity. The first oracle attempt incorrectly used
the condition mean; correcting the comparison preserved the tolerance.
The charged small case checks per-box mass and charge conservation and
non-increasing particle number. These are numerical checks, not measurements
of stochastic convergence of the full-size collision counts.

A pre-commit 100-step full-layout pilot with the same physics completed 0.0001 s on CUDA with 2,097,152 initial
particles. Synchronized stepping took 83.80 s; preparation/capture took
1.87 s. Observed GPU memory usage was approximately 545 MiB (device-wide
snapshot, not a peak allocation measurement). Maximum per-box species-mass
relative residual was 4.108e-15. This pilot selected the 1000-step publication
workload; it is not a substitute for that full run.

## Complete publication run

```bash
WARP_CACHE_PATH=/tmp/particula-charged-warp .venv/bin/python -u \
  scripts/docs_examples.py refresh gpu-environment-sweep
```

The first complete generation, `run-20261114T145528707213Z`, succeeded in
842.26 s overall (836.64 s synchronized stepping; 1.67 s preparation/capture).
It ended with 2,097,028 particles after 124 accepted mergers. Maximum per-box
species-mass relative residual was 1.688e-14; every charge check was exact,
and final masses were finite and nonnegative. Its source and helper hashes
match the recorded revision above. Installed distribution metadata remains
0.2.14 while the selected source is 0.3.0; the environment was not upgraded.

| T [K] | P [Pa] | Initial RH | Final mean RH | Mergers |
| --- | --- | --- | --- | --- |
| 278.15 | 70000 | 0.40 | 0.400020 | 10 |
| 278.15 | 70000 | 0.60 | 0.599183 | 10 |
| 278.15 | 70000 | 0.80 | 0.796846 | 8 |
| 278.15 | 70000 | 0.90 | 0.893270 | 9 |
| 278.15 | 101325 | 0.40 | 0.400020 | 8 |
| 278.15 | 101325 | 0.60 | 0.599183 | 6 |
| 278.15 | 101325 | 0.80 | 0.796836 | 9 |
| 278.15 | 101325 | 0.90 | 0.893146 | 7 |
| 298.15 | 70000 | 0.40 | 0.400006 | 10 |
| 298.15 | 70000 | 0.60 | 0.599757 | 7 |
| 298.15 | 70000 | 0.80 | 0.799041 | 12 |
| 298.15 | 70000 | 0.90 | 0.897748 | 1 |
| 298.15 | 101325 | 0.40 | 0.400006 | 5 |
| 298.15 | 101325 | 0.60 | 0.599757 | 9 |
| 298.15 | 101325 | 0.80 | 0.799041 | 7 |
| 298.15 | 101325 | 0.90 | 0.897743 | 6 |

These are single-realization counts, not significance tests. Actual figure
inspection found crowded pressure labels in the first collision chart.
The revised source uses horizontal bars; a second complete refresh is pending
with identical physics, workload and seed. The first generation is retained.

## Publication checks

The documentation tooling tests (25 Python tests and three Node tests),
documentation structure check and changed-file Ruff checks passed. The first saved
bundle passed integrity checks and the site built 456 pages / 586 HTML files
with zero link errors. Final plot-refresh integrity and extracted ZIP execution
remain pending. All nine study tests passed again on the final helper source. The library itself is unchanged by this example;
its broader validation remains in the
[charged coagulation record](charged-coagulation.md).
