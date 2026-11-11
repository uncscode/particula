# M3 decision: preserve PMF/PDF coagulation and document its limitation

On 2026-11-11 the maintainer explicitly answered the pending scope question:
**“Preserve method; document limitation.”** This decision supersedes the
moment-transport extension in `450e8dd68` and the original requirement for
arbitrary per-species conservation in the density-step row. It does not waive
resolved CPU/GPU conservation or change condensation and wall-loss contracts.

## Current implementation

- Restore the existing explicit PMF/PDF number-density update and fixed
  per-bin masses/charge. Only concentration is a density-step destination.
  Validate current physical state, staging, aliases and writable destinations;
  reject negative/nonfinite proposals before publication, without clipping.
- Remove additive species/charge-moment transport. Read-only masses/charge
  remain admissible protected input for these density steps.
- Preserve resolved selectors, weight restrictions, volume normalization,
  species/charge conservation, GPU paths and all other M3 process work.

The fixed composition at a destination radius generally differs from the
combined donor composition. For size-dependent composition, each species and
weighted charge need not be conserved even when total-mass error converges.
Grid refinement and smaller timesteps do not eliminate that composition
defect. Conservative species/charge transport is deferred; this is not an M4
implementation commitment or a general scientific-acceptance decision.

The maintainer further clarified intended use: uniform composition across
radius, or a single effective-material/effective-density representation. The
method is generally not appropriate for tracking composition varying with
radius. Effective density is a bulk approximation and does not supply missing
species transport. This guidance does not introduce a new runtime rejection
or claim that uniform composition eliminates discretization error.

The reproducible equal-density, constant-kernel lognormal fixture in
`native_coagulation_limitations_test.py` independently integrates raw mass
times population on 128/256/512/1024 nodes. Its PDF species defects approach
+7.79% and -12.20% of gross transferred species mass, while the total defect
decreases. PMF at 1024 nodes has +7.43% and -12.49% species defects. These are
fixture-specific results, not permitted-error tolerances or fractions of the
starting inventory. Full fixture parameters and earlier values remain in
`p4_validation.md` and the user-facing coagulation feature contract.

## Assertion migration

- Restore the original fixed-mass/fixed-charge assertions in
  `test_density_step_equals_existing_rate_times_duration`.
- Replace the 49 extension-specific cases in
  `native_coagulation_moments_test.py` with 26 preserved-method cases:
  unchanged number law and all array identities for ten mechanisms and both
  representations, read-only fixed-material admission, and an independent
  refinement characterization of the accepted species defect.
- Update the six density integration cases to assert the existing number law,
  fixed material/charge, explicit inventory calculation and subsequent wall
  sink. Their former moment-conservation bound belongs to the removed extension.
- Retain all resolved conservation tests and pre-extension density rejection,
  ownership and uniform-composition refinement assertions. No conserved-quantity
  tolerance was relaxed to pass this decision; the density contract explicitly
  acknowledges nonclosure instead of claiming conservation.

`p4_moments_validation.md` is marked historical and superseded. P6's previous
test totals describe the former head and are not evidence for this revision.

## Validation

Changed coagulation and native integration checks:

```bash
.venv/bin/pytest particula/dynamics/coagulation/ \
  particula/integration_tests/native_scientific_processes_test.py \
  -q --no-cov --tb=short
.venv/bin/python .opencode/tools/run_pytest.py
.venv/bin/python .opencode/tools/run_linters.py --no-auto-fix
.venv/bin/ruff format --check particula/
.venv/bin/mkdocs build --strict
```

Focused result: 786 passed in 3.89 s. Untargeted package gate: **9,046 passed,
19 skipped in 154.72 s; 92% coverage**, unchanged 80% threshold. Repository
Ruff/mypy passed; all 536 Python files are formatted. Strict docs passed in
1.42 s and were rebuilt after recording these results. Logs:
`.artifacts/e9-m3/preserved-method/`. The reduced count reflects replacement
of the removed extension's 49 cases with 26 preserved-method cases. The
previous separate wall-loss, example and infrastructure evidence is unchanged;
GitHub's final head checks supply the subsequent remote matrix results.
