# Mass Precision Recommendation Report

This page is the final `E2-F6` acceptance artifact for particle mass precision
policy. It records the shipped P1-P3 evidence, publishes the current
recommendation for downstream dtype/schema work, and keeps measured evidence
separate from follow-up constraints.

Production particle storage remains absolute per-species `np.float64` on CPU
and `wp.float64` on GPU. This report does not ship a runtime dtype or schema
change.

## Final recommendation

- Keep absolute per-species `np.float64` particle masses on CPU.
- Keep absolute per-species `wp.float64` particle masses on GPU/Warp mirrors.
- Treat this baseline as the accepted production policy until a later proposal
  proves that an alternative representation preserves the documented fidelity,
  conservation-sensitive behavior, and workflow constraints in this report.

## Recommendation boundaries

- This report summarizes only evidence already executed in the shipped P1-P3
  test surface.
- The recommendation is an approval gate for downstream dtype/schema proposals,
  not approval to change production defaults now.
- Optional benchmark evidence remains supplemental and opt-in; it does not by
  itself authorize a production migration.

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

## Shipped evidence summary

The shipped evidence comes from the deterministic baseline fixture in
`particula/gpu/tests/mass_precision_cases_test.py`, the comparison suite in
`particula/gpu/tests/mass_precision_metrics_test.py`, and the fast benchmark
helper coverage in `particula/gpu/tests/benchmark_helpers_test.py`.

### P1 baseline coverage

- Deterministic cases span NPF-scale through droplet-scale particles:
  `npf_cluster`, `five_to_ten_nm`, `accumulation_mode`, and `cloud_droplet`.
- All baseline inputs use explicit `np.float64` arrays and canonical container
  shapes.
- Mixed-species baselines reconstruct expected radii from authored total volume,
  fixed species volume fractions, and documented densities.

### P2 executed candidates

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

### P3 thresholds and conservation-sensitive checks

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
  - enforce aggregate species-total relative tolerance
    `_AGGREGATE_DELTA_RTOL = 5e-7` separately from per-particle checks

The fast P3 suite keeps representative coverage bounded to the shipped four
deterministic cases plus one mixed-scale stress case where nanometer particles
and droplet-scale particles coexist in the same
`(n_boxes, n_particles, n_species)` array.

### P3 mixed-scale fidelity finding

The mixed-scale stress case uses a single-species deterministic array that puts
`1.5e-9 m` particles in the same boxes as `8.0e-6 m` to `1.5e-5 m` droplets.
The executable review threshold is intentionally focused on the smallest
particle slice so whole-array aggregates cannot hide fragile small-particle
loss.

Aggregate mixed-scale checks are tracked separately from the smallest-particle
slice checks. The aggregate assertion applies to species totals over the full
array, while the mixed-scale smallest-particle assertions remain the guardrail
for nanometer-scale fidelity.

### P3 zero-mass and zero-volume edge handling

- Zero-total-mass reconstruction remains deterministic for
  `fp32_total_mass_fp32_mass_fraction`:
  - projected total mass is exactly zero
  - projected mass fractions are exactly zero
  - reconstructed masses and derived radii are exactly zero
- Zero-volume / zero-effective-radius paths are exercised with explicit zero
  particle masses and zero mass-transfer rates.
- The fast metric suite is rerun under `pytest -Werror` to keep these paths
  divide-by-zero-warning free.

### P3 clamp accounting

When evaporation-oriented comparisons would drive raw updated mass below zero,
the study records three separate quantities:

1. `raw_updated_mass = initial_mass + raw_mass_transfer`
2. `post_clamp_mass = maximum(raw_updated_mass, 0.0)`
3. `clamp_delta = post_clamp_mass - raw_updated_mass`

The executable metrics also report clamp frequency as the number of entries
where `raw_updated_mass < 0.0`, and aggregate clamp delta per species. This
keeps raw reconstruction error distinct from clamp-induced mass changes.

### P3 memory-footprint examples

These examples use analytic `shape × dtype-size` accounting only.

| Candidate | Example shape | Formula | Approx. bytes |
| --- | --- | --- | ---: |
| `fp32_absolute_mass` | `10 × 100,000 × 3` masses | `10 * 100000 * 3 * 4` | `12,000,000` |
| `mixed_precision_mass_plus_density` | masses + concentration + charge + volume + density with `10` boxes, `100,000` particles, `3` species | `(10 * 100000 * 3 * 4) + (10 * 100000 * 4) + (10 * 100000 * 4) + (10 * 4) + (3 * 8)` | `20,000,064` |
| `fp32_total_mass_fp32_mass_fraction` | total mass + mass fractions with `10 × 100,000 × 3` | `(10 * 100000 * 4) + (10 * 100000 * 3 * 4)` | `16,000,000` |
| Baseline `fp64` masses | `10 × 100,000 × 3` masses | `10 * 100000 * 3 * 8` | `24,000,000` |

These examples document storage tradeoffs only. They do not change production
runtime schemas or default dtypes.

### P3 throughput evidence availability

- Fast default validation remains in
  `particula/gpu/tests/mass_precision_metrics_test.py`.
- Optional throughput evidence lives on the existing benchmark surface in
  `particula/gpu/tests/benchmark_test.py`.
- The P3 benchmark path records bounded study-only candidate-payload timings for
  three representative case/candidate pairs, including the full documented
  projection payload for each candidate rather than reconstruction-only mass
  casts, and still requires the explicit `--benchmark` opt-in plus CUDA
  availability.
- On machines without Warp or CUDA support, the benchmark module skips cleanly
  for GPU execution, while CPU-only helper coverage remains importable and the
  fast metric suite remains runnable.

## Executed but not recommended candidates

- `fp32_absolute_mass`
- `mixed_precision_mass_plus_density`
- `fp32_total_mass_fp32_mass_fraction`

These candidates were executed to measure reconstruction fidelity,
conservation-sensitive deltas, mixed-scale behavior, clamp accounting, and
storage tradeoffs. They are not recommended for production defaults in this
phase because the shipped evidence is used to approve the current `fp64`
baseline, not to authorize a runtime migration.

## Unsupported candidates

Candidates that need new runtime schema fields, extra production metadata, or
public API expansion remain unsupported in this phase. They should be recorded
as documentation-only ideas rather than added to executable runtime code or the
focused test matrix.

## Deferred investigation areas

- Broader alternative mass representations such as log-mass or new reference-
  mass schemas.
- Production schema expansion that introduces new stored helper fields or new
  public migration obligations.
- Wider throughput campaigns beyond the focused opt-in benchmark surface.

These follow-up areas are out of scope for this issue.

## Downstream constraints for future dtype/schema proposals

Any future proposal that changes production defaults must, at minimum:

- preserve the deterministic P1 case coverage from NPF to droplet scale,
- satisfy the shipped P3 reconstruction and mixed-scale thresholds with exact
  candidate ids and warning-clean validation,
- demonstrate acceptable conservation-sensitive mass-transfer deltas,
- account for clamp behavior explicitly rather than hiding it inside aggregate
  error metrics,
- document storage/memory tradeoffs truthfully, and
- update downstream roadmap and migration guidance in the same change so the
  canonical policy reference stays consistent.

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

## Publication-readiness validation

Rerun the focused commands below before treating this report as the backing
reference for downstream dtype/schema work:

- `pytest particula/gpu/tests/mass_precision_cases_test.py -q`
- `pytest particula/gpu/tests/mass_precision_metrics_test.py -q`
- `pytest particula/gpu/tests/benchmark_helpers_test.py -q`
- `pytest -Werror particula/gpu/tests/mass_precision_metrics_test.py -q`

Optional throughput evidence remains opt-in only:

- `pytest particula/gpu/tests/benchmark_test.py --benchmark -k mass_precision -v -s`

Check every Markdown link added by this report update directly and run
`mkdocs build --strict` when the docs toolchain is available.

## Reproduction commands

```bash
pytest particula/gpu/tests/mass_precision_cases_test.py -q
pytest particula/gpu/tests/mass_precision_metrics_test.py -q
pytest particula/gpu/tests/benchmark_helpers_test.py -q
pytest -Werror particula/gpu/tests/mass_precision_metrics_test.py -q
pytest particula/gpu/tests/benchmark_test.py --benchmark -k mass_precision -v -s
```
