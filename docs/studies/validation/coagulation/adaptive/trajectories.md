# Adaptive CUDA coagulation trajectories

See the [qualification record](../direct-coagulation/qualification.md) for the associated implementation decision and reproduction gates.

## Adaptive outer intervals on native CUDA

Two synthetic, unit-weight cloud populations reached 600 s in six 100 s
observation intervals. Both used seed 17, 283.15 K, 101325 Pa, combined
Brownian, SP2016 settling and ST1956 shear, dissipation 10⁻⁴ m² s⁻³ and
fluid density 1.2 kg m⁻³. Total concentration was 2,000 cm⁻³, with 10% of
the slots assigned to droplets. Each run exhausted a one-pass limit,
checkpointed the committed state, and resumed with a 128-pass ceiling. Each
accepted pass was limited to 256 scheduled trials per box. These two
resolutions exercise the issue's mechanisms and trajectory length. The
original campaign comprised 300 cases across a wider concentration range.
[Worked CUDA example and saved results](adaptive_coagulation.py)

| Initial slots | Accepted passes in successive 100 s intervals | Mergers | Live + rollback bytes | Synchronous step time |
| ---: | --- | ---: | ---: | ---: |
| 5,000 | 8, 9, 10, 10, 10, 10 | 184 | 503,876 + 503,876 | 0.56–0.68 s |
| 10,000 | 16, 18, 18, 19, 22, 22 | 433 | 964,516 + 964,516 | 4.13–5.35 s |

Each final active count equalled initial count minus accepted mergers.
Survivors retained unit weight, charge remained zero and both relative
mass drifts rounded to zero in fp64. Times include the restore-point copies,
graph replay and status readback. An additional status check immediately
after each committed step took 105–142 µs on the idle stream. This measures
host observation overhead separately from time waiting for pending physics.
The saved run used revision `410d6bc858`, an RTX 5060, Warp 1.11.1,
Python 3.12.12 and NumPy 2.3.4.

A separate two-box, three-material fixture isolates evolving disjoint-pair
capacity. At seed zero, a fixed 0.1 s step passes in both boxes. After the
first merger in box zero, five active particles permit only two disjoint
trials. Its live majorant is 2.5088956262 × 10⁻¹⁵ m³ s⁻¹ on both Warp
CPU and CUDA. Both boxes need three trials at 0.1 s and complete the
interval in two admitted 0.05 s passes.

The focused tests compare complete particle, field, collision, RNG and
checkpoint state after a rejected composed step. They check ordinary versus
captured continuation, ordered direct collisions against manual replay,
independent work limits, ULP boundaries, nonfinite demand and cohort
accounting. A separate Warp CPU comparison uses 32 independent seeds per
arm over 0.2 s. The adaptive 0.1 s and fixed 0.0125 s arms compare merger
counts and mean squared active radius against four combined standard errors.
The adaptive and refined arms averaged 5.28 ± 0.19 and 4.88 ± 0.24 mergers,
respectively. Their mean squared active radii were 1.077 ± 0.021 and
1.041 ± 0.032 µm² (mean ± one standard error across seeds). Both
differences fall within the specified band for this small fixture.

```bash
python -m pytest particula/execution/tests/coagulation_subcycling_test.py -q -Werror
python scripts/docs_examples.py refresh adaptive-coagulation
```
