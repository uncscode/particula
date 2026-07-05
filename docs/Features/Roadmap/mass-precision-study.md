# Mass Precision Study Baseline and Candidate Fidelity Checks

This page records the deterministic baseline cases and study-only candidate
fidelity checks for the particle mass precision study used by `E2-F6`. In this
phase, production particle storage remains absolute per-species `np.float64` on
CPU and `wp.float64` on GPU.

## Baseline policy

- CPU particle arrays remain absolute per-species `np.float64`.
- Warp particle mirrors remain absolute per-species `wp.float64`.
- This phase defines baseline cases plus bounded candidate fidelity checks.
- This phase does not recommend a future precision or schema migration.

## Deterministic cases

The test fixture in
`particula/gpu/tests/mass_precision_cases_test.py` defines the following cases
in ascending radius order:

1. `npf_cluster` — new-particle-formation cluster, target radius `1.5e-9 m`
2. `five_to_ten_nm` — 5-10 nm particle, target radius `7.0e-9 m`
3. `accumulation_mode` — accumulation mode, target radius `1.5e-7 m`
4. `cloud_droplet` — cloud droplet, target radius `1.0e-5 m`

Each case uses deterministic `np.float64` arrays with the canonical particle
storage shapes:

- `masses -> (n_boxes, n_particles, n_species)`
- `concentration -> (n_boxes, n_particles)`
- `charge -> (n_boxes, n_particles)`
- `density -> (n_species,)`
- `volume -> (n_boxes,)`

## Density assumptions

- `npf_cluster`: `1000.0 kg/m^3`
- `five_to_ten_nm`: `1100.0 kg/m^3`
- `accumulation_mode`: species densities `[1200.0, 1800.0] kg/m^3`
- `cloud_droplet`: species densities `[1000.0, 1770.0] kg/m^3`

## Volume-fraction assumptions

- `npf_cluster`: `[1.0]`
- `five_to_ten_nm`: `[1.0]`
- `accumulation_mode`: `[0.65, 0.35]`
- `cloud_droplet`: `[0.92, 0.08]`

## Construction rule

All baseline masses use the same spherical mass formula inverted by
`ParticleData.radii`:

```text
mass = (4.0 / 3.0) * π * radius^3 * density
```

Mixed-species cases split the implied total particle volume across fixed species
volume fractions before converting each species volume to mass.

The regression suite reconstructs the expected radii for every deterministic
case, including the mixed-species `accumulation_mode` and `cloud_droplet`
baselines, from the authored total particle volume implied by those masses and
densities.

All reported radii use meters, densities use `kg/m^3`, and volume fractions are
unitless fractions.

## Executed P2 candidates

The focused comparison module `particula/gpu/tests/mass_precision_metrics_test.py`
evaluates exactly three study-only candidates against the `fp64` baseline:

1. `fp32_absolute_mass`
   - Store per-species masses as `np.float32`.
   - Reconstruct comparable analysis inputs by casting those masses back to
     `np.float64`.
2. `mixed_precision_mass_plus_density`
   - Keep `density` at `np.float64`.
   - Store `masses`, `concentration`, `charge`, and `volume` as `np.float32`.
   - Reconstruct comparable analysis inputs by casting the candidate-side
     arrays back to `np.float64` before checking mass and radius fidelity.
3. `fp32_total_mass_fp32_mass_fraction`
   - Store per-particle total mass and per-species mass fractions as
     `np.float32`.
   - Reconstruct per-species masses in `np.float64` as
     `total_mass[..., None] * mass_fractions`.
   - Zero-total-mass particles reconstruct deterministically to zeros rather
     than relying on divide-by-zero warnings.

For all three candidates, the study compares reconstructed per-species masses
and derived radii against the baseline `np.float64` fixture values with bounded
`numpy.testing.assert_allclose` tolerances.

## P3 conservation and reconstruction thresholds

Phase P3 extends the same case-candidate matrix with cached executable metrics:

- pure reconstruction error:
  - relative mass error threshold: `5e-7`
  - relative radius error threshold: `2e-7`
- mixed-scale smallest-particle stress case:
  - relative mass error threshold: `6e-7`
  - relative radius error threshold: `2.5e-7`
- CPU-reference mass-transfer comparison:
  - use `particula.dynamics.condensation.mass_transfer.get_mass_transfer`
  - build deterministic `mass_rate`, `time_step`, `gas_mass`,
    `particle_mass`, and `particle_concentration` directly from the cached
    baseline and reconstructed arrays
  - report both per-particle deltas and aggregate species-total deltas

The fast P3 suite keeps representative coverage bounded to the shipped four
deterministic cases plus one mixed-scale stress case where nanometer particles
and droplet-scale particles coexist in the same
`(n_boxes, n_particles, n_species)` array.

## P3 mixed-scale fidelity finding

The mixed-scale stress case uses a single-species deterministic array that puts
`1.5e-9 m` particles in the same boxes as `8.0e-6 m` to `1.5e-5 m` droplets.
The executable review threshold is intentionally focused on the smallest
particle slice so whole-array aggregates cannot hide fragile small-particle
loss.

## P3 zero-mass and zero-volume edge handling

- Zero-total-mass reconstruction remains deterministic for
  `fp32_total_mass_fp32_mass_fraction`:
  - projected total mass is exactly zero
  - projected mass fractions are exactly zero
  - reconstructed masses and derived radii are exactly zero
- Zero-volume / zero-effective-radius paths are exercised with explicit zero
  particle masses and zero mass-transfer rates.
- The fast metric suite is rerun under `pytest -Werror` to keep these paths
  divide-by-zero-warning free.

## P3 clamp accounting

When evaporation-oriented comparisons would drive raw updated mass below zero,
the study records three separate quantities:

1. `raw_updated_mass = initial_mass + raw_mass_transfer`
2. `post_clamp_mass = maximum(raw_updated_mass, 0.0)`
3. `clamp_delta = post_clamp_mass - raw_updated_mass`

The executable metrics also report clamp frequency as the number of entries
where `raw_updated_mass < 0.0`, and aggregate clamp delta per species. This
keeps raw reconstruction error distinct from clamp-induced mass changes.

## P3 memory-footprint examples

These examples use analytic `shape × dtype-size` accounting only.

| Candidate | Example shape | Formula | Approx. bytes |
| --- | --- | --- | ---: |
| `fp32_absolute_mass` | `10 × 100,000 × 3` masses | `10 * 100000 * 3 * 4` | `12,000,000` |
| `mixed_precision_mass_plus_density` | masses + concentration + charge + volume + density with `10` boxes, `100,000` particles, `3` species | `(10 * 100000 * 3 * 4) + (10 * 100000 * 4) + (10 * 100000 * 4) + (10 * 4) + (3 * 8)` | `20,000,064` |
| `fp32_total_mass_fp32_mass_fraction` | total mass + mass fractions with `10 × 100,000 × 3` | `(10 * 100000 * 4) + (10 * 100000 * 3 * 4)` | `16,000,000` |
| Baseline `fp64` masses | `10 × 100,000 × 3` masses | `10 * 100000 * 3 * 8` | `24,000,000` |

These examples document storage tradeoffs only. They do not change production
runtime schemas or default dtypes.

## P3 throughput evidence availability

- Fast default validation remains in
  `particula/gpu/tests/mass_precision_metrics_test.py`.
- Optional throughput evidence lives on the existing benchmark surface in
  `particula/gpu/tests/benchmark_test.py`.
- The P3 benchmark path records bounded study-only projection timings for three
  representative case/candidate pairs and still requires the explicit
  `--benchmark` opt-in plus CUDA availability.
- On machines without Warp or CUDA support, the benchmark module skips cleanly
  and the fast metric suite remains runnable.

## Unsupported candidates

Candidates that need new runtime schema fields, extra production metadata, or
public API expansion remain unsupported in this phase. They should be recorded
as documentation-only ideas rather than added to executable runtime code or the
focused test matrix.

## Reproducibility

- No random draws are used.
- Arrays are rebuilt from fixed constants.
- Case-candidate projections and CPU-reference mass-transfer comparisons are
  cached once per deterministic input pair and then reused across assertions.
- Tests assert exact rerun stability and exact Warp CPU-device round trips for
  representative single-species and mixed-species cases.
- Tests also assert finiteness, nonnegative masses, and malformed-input
  rejection.

## Documentation cross-check checklist

- Candidate names in this page match the executable candidate ids.
- Threshold values match the fast metric assertions.
- Memory examples use the documented formulas and concrete shapes.
- Reproduction commands cover both fast checks and optional benchmark evidence.

## Reproduction commands

```bash
pytest particula/gpu/tests/mass_precision_cases_test.py -q
pytest particula/gpu/tests/mass_precision_metrics_test.py -q
pytest -Werror particula/gpu/tests/mass_precision_metrics_test.py -q
pytest particula/gpu/tests/benchmark_test.py --benchmark -k mass_precision -v -s
```
