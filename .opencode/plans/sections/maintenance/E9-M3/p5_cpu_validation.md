# P5 native CPU wall-loss checkpoint

This is the historical CPU checkpoint. Subsequent GPU evidence is in
`p5_gpu_validation.md`; final integration and gates are in `p6_validation.md`.

Entry revision: `450e8dd68`. This checkpoint implements the CPU portion of
P5 in the single M3 PR. Direct/prepared GPU wall-loss admission and sink tests
remain pending; P5/M3 completion and M4 admission are not claimed.

## Delivered contract

- Native coefficient, loss-rate, rate and step dispatch for neutral spherical,
  neutral rectangular, charged spherical and charged rectangular strategies.
  Inputs are explicit single-box `ParticleData`, not facade reconstructions.
  PMF/PDF retain density storage at V=1; resolved rates use raw count/V while
  removal updates the original counts directly. Unequal resolved counts are
  supported because each representative is wholly retained or removed.
- Revalidate current representation, material/charge state, derived geometry
  and density, environment, mutable geometry/field controls and finite time.
  Existing physical coefficient helpers and charged limiting remain unchanged.
  Native charged evaluation uses current explicit charge arrays, never the
  legacy cached-charge convenience state.
- Density stepping preserves the CPU explicit Euler law. Reject nonfinite or
  negative proposals before mutation rather than changing the law to an
  exponential update or introducing clipping.
- Resolved stepping preserves the CPU Bernoulli survival law. Neutral helper
  draws include positive-radius unoccupied material carriers; charged draws
  use occupied positive-radius slots. Native zero-time/wholly-unoccupied calls
  validate but draw nothing. RNG rollback after drawing is not promised.
- Resolved loss clears selected masses/counts, retains CPU charge storage,
  and preserves survivors and pre-existing inactive material. Consequently
  removed charged slots do not automatically satisfy the separate activation
  free-slot contract. M4 composition must account for this existing CPU
  distinction; GPU slot clearing is not substituted here.
- Stage updates before publication, retain container/array identities, and
  reject unwritable/overlapping destinations, including aliases with array-
  backed geometry/field configuration. Density/volume/PDF grid and charge
  storage are protected. Public step overloads retain the exact native or
  legacy carrier return type for existing callers, without M4 runnable edits.

## Assertion migration and scientific evidence

Both named suites now construct data directly through
`dynamics/tests/wall_loss_native_helpers.py`. All original test functions and
their geometry, rates, shape, empty-state, field/image-charge, helper-parity,
survival and stochastic assertions remain. Continuous cases now have their
own explicitly tagged PDF fixtures rather than reusing an untagged PMF facade.
No fixture uses `from_representation` or a legacy preset builder.

`dynamics/tests/native_wall_loss_test.py` adds 385 tests:

- Independent forward-Euler density updates and removed-species inventory;
  PDF inventory is integrated explicitly by radius trapezoids.
- Controlled Bernoulli draws at V=.25/1/4, unequal fractional/multiple counts,
  independent extensive/density species sinks and weighted-charge balance,
  survivor and inactive-state invariants, and exact raw-count updates.
- Independent weighted survival statistics: mean `p*sum(weights)`, variance
  `p*(1-p)*sum(weights**2)`, with six-sigma bounds for 12,000 representatives.
  This is not a cross-backend exact-seed comparison.
- Independently derived mass/volume radii and mass-weighted densities supplied
  to unchanged coefficient laws, including explicit charged state despite a
  stale legacy cache. This verifies native input plumbing, not an independent
  rederivation of every existing electrostatic/transport equation; both
  migrated helper-parity suites retain that lower-level regression evidence.
- Mutated fields/controls, invalid environment/time/coefficient output,
  n_boxes!=1, distribution mismatch, nonunit PMF/PDF volume, occupied zero mass,
  aliases/read-only storage, zero-work RNG preservation and empty capacity.
- Valid array-backed rectangular geometry/field configuration matches tuple
  configuration and remains caller-owned. Final review corrected an ambiguous
  ndarray truth check in the shared geometry-scale helper to a None check.
  Constructors still normalize their geometry inputs as before.

Temporary facade dispatch remains for M4/M5 consumers. Existing runnable
regressions still exercise it; deletion remains M6-owned.

## Validation

Using the repository `.venv`:

- Native wall-loss module: 385 passed in 0.27 s.
- `pytest particula/dynamics/tests/wall_loss_strategies_test.py -q --no-cov`:
  16 passed in 0.03 s on the final CPU revision.
- `pytest particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py -q
  --no-cov`: 43 passed in 0.06 s, in a separate invocation. This directory is
  recursively excluded and is not represented by the untargeted package gate.
- Explicit adjacent builder/factory files: 28 passed in 0.04 s.
- Earlier normal/native/runnable combined checkpoint: 337 passed in 2.06 s,
  before the final 74 admission assertions were added.
- Repository lint/type runner passed after correcting the exact-carrier return
  annotation; final changed-file Ruff and format checks passed.
- Strict documentation build passed in 1.63 s.
- Initial untargeted package runner: 8,803 passed, 19 skipped in 152.84 s,
  92% coverage with unchanged 80% threshold. This preceded the final geometry
  None-check correction and its added regression.
- Final untargeted package runner against `450e8dd68` plus this CPU P5 diff:
  8,804 passed, 19 skipped in 152.23 s; 92% coverage, unchanged 80% threshold.
  `.artifacts/e9-m3/p5-cpu/package-tests-final.log` is the final CPU checkpoint
  result. It does not supply the still-pending native GPU P5 evidence.
  Logs and focused results are under `.artifacts/e9-m3/p5-cpu/`.

The initial new-test failures were invalid fixture volume dtype and an occupied
zero-mass lane in an alias-specific test. Both were corrected in the test data;
the physical admission checks were retained. No numerical tolerances or
coverage thresholds were relaxed.
