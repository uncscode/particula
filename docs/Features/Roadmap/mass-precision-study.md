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
`numpy.testing.assert_allclose` tolerances. The study remains limited to the
shipped four deterministic cases and does not add throughput benchmarks,
aggregate tradeoff tables, or a final migration recommendation.

## Unsupported candidates

Candidates that need new runtime schema fields, extra production metadata, or
public API expansion remain unsupported in this phase. They should be recorded
as documentation-only ideas rather than added to executable runtime code or the
focused test matrix.

## Reproducibility

- No random draws are used.
- Arrays are rebuilt from fixed constants.
- Tests assert exact rerun stability and exact Warp CPU-device round trips for
  representative single-species and mixed-species cases.
- Tests also assert finiteness, nonnegative masses, and malformed-input
  rejection.

## Reproduction commands

```bash
pytest particula/gpu/tests/mass_precision_cases_test.py -q
pytest particula/gpu/tests/mass_precision_metrics_test.py -q
pytest -Werror particula/gpu/tests/mass_precision_metrics_test.py -q
```
