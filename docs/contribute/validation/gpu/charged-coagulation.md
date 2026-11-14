# Charged coagulation and atmospheric ion validation

Development evidence, 2026-11-13, branch
`codex/gpu-charged-coagulation-lucky-few`, based on `19a8fb11d`.
These results concern the implementation on this branch; no release is claimed.

## Implemented boundary

The five existing charged models can run directly or through prepared Warp
execution. A bounded parallel reduction supplies the exact pair maximum to the
existing stochastic sampler. Prepared CUDA capture and same-device checkpoint
continuation retain the selected model. Existing Brownian defaults remain.

The CPU Dyachkov dimensionless formula now evaluates differences with `expm1`
and an equivalent rearrangement to avoid cancellation near neutrality. The
analytic neutral limit is checked independently; CUDA agreement does not define
that limit. CUDA decimal fit coefficients are retained as explicit fp64
constants rather than rounding through float32 literals.

`IonPairSource` injects explicit positive/negative particles with unit counts,
independent persistent source randomness, a fixed-capacity rejection boundary,
and cumulative external-mass accounting. `IonNeutralCleanup` exports only
neutral particles made entirely from both reserved ion materials, with persistent
species mass and particle ledgers. Neither establishes ion collision physics or
the completed Lucky Few survival study.

The LF2013 helper rejects unsupported domains by default. Optional lower/upper
values for omitted repulsive fluxes are uncertainty bounds, not an extrapolated
fit or an atmospheric validation result. Same-sign singly charged seed particles
with diameter 3 nm fall below the corresponding fit's lower radius, approximately
5.308 nm. Production treatment remains a scientific decision for the full study.

## Device and checks

Host CUDA: NVIDIA GeForce RTX 5060, 8151 MiB; Warp 1.11.1; CUDA toolkit 12.9,
driver 13.4. The restricted sandbox cannot see the driver; CUDA qualification
was performed with host access. No CPU fallback was used for CUDA cases.

Commands below used `WARP_CACHE_PATH=/tmp/particula-charged-warp` where relevant.

| Check | Result |
| --- | --- |
| `.venv/bin/pytest particula/gpu/kernels/tests/coagulation_test.py particula/execution/tests/prepared_test.py particula/execution/tests/prepared_restart_capture_test.py -q --no-cov -Werror` | 942 passed, host CPU/CUDA |
| `.venv/bin/pytest particula/gpu/dynamics/tests/charged_models_test.py particula/execution/tests/prepared_charged_test.py particula/dynamics/ions/tests particula/dynamics/coagulation/tests/charged_dimensionless_kernel_test.py -q --no-cov -Werror` | 43 passed, host CPU/CUDA |
| `.venv/bin/pytest particula/execution/tests/prepared_ions_test.py particula/gpu/kernels/tests/ion_source_test.py particula/execution/tests/prepared_restart_capture_test.py particula/execution/tests/prepared_charged_test.py -q --no-cov -Werror` | 62 passed, host CPU/CUDA |
| `.venv/bin/pytest particula -Werror --cov=particula --cov-report=term-missing --cov-fail-under=80` | 10,047 passed, 14 skipped, 1 xfailed; 92.04% coverage; host CPU/CUDA |
| `.venv/bin/pytest examples_tests scripts/tests -q --no-cov -Werror` | 459 passed, 1 skipped; host execution |
| `.venv/bin/python scripts/docs_examples.py refresh gpu-charged-models` | Complete CUDA run; saved figure visually inspected |
| `.venv/bin/python -u scripts/docs_examples.py check` | Passed; all saved bundles current and CPU-policy examples executed; CUDA/manual policies require explicit refresh |
| `.venv/bin/ruff check particula/ examples_tests/ scripts/ --no-fix` | Passed |
| `.venv/bin/ruff format particula/ examples_tests/ scripts/ --check` | Passed |
| `.venv/bin/mypy particula/ --ignore-missing-imports` | Passed |
| `.venv/bin/python scripts/check_docs_structure.py` | 0 errors |
| `npm --prefix website test` | 3 passed |
| `npm --prefix website run build` | Passed; 0 link errors |

The full package run at `80e271627` includes paired-source integration,
composition and zero-production tests. Subsequent cleanup changes are covered
by the focused checks below; that earlier full run does not qualify them.
The existing functional and object charged examples
were also refreshed; all four saved figures were visually inspected. Notebook
execution required host access; restricted notebook processes were stopped
after stalling during kernel startup. A restricted tooling run also hit
the read-only Warp cache; the successful host rerun above supersedes it.

Hosted CI is separate from these local results. All six workflows passed at
`8d8f476fd`; newer increments require their own hosted results.
No speedup or atmospheric survival probability is established by this record.

## Neutral-product cleanup follow-up

Cleanup tests check that neutral aerosols remain, charged ion-only particles
remain, each exported species is accounted for, repeated cleanup is a no-op,
and ledger/state continuation survives checkpoint and native CUDA capture.
Invalid reserved indices reject at preparation; evolving fractional counts
fault before modifying the affected box. The source and cleanup retain distinct
families so the example must explicitly choose their ordering.

Host command: `WARP_CACHE_PATH=/tmp/particula-charged-warp .venv/bin/pytest
particula/execution/tests/prepared_ion_cleanup_test.py
particula/execution/tests/prepared_ions_test.py
particula/execution/tests/prepared_restart_capture_test.py -q --no-cov -Werror`:
58 passed, including actual CUDA. Ruff and mypy passed (571 source files).

## Original-seed lineage follow-up

`SeedCohort` stores unresolved original-lineage counts and persistent background
labels separately from particle inventories. It consumes only applied compacted
collision pairs and observes size after each physical process. Outcome counts
partition the original cohort into reached, captured and unresolved lineages.

The affected prepared suite passed 171 tests on host CPU/CUDA:

```bash
WARP_CACHE_PATH=/tmp/particula-charged-warp .venv/bin/pytest \
  particula/execution/tests/prepared_cohort_test.py \
  particula/execution/tests/prepared_test.py \
  particula/execution/tests/prepared_charged_test.py \
  particula/execution/tests/prepared_ions_test.py \
  particula/execution/tests/prepared_ion_cleanup_test.py \
  particula/execution/tests/prepared_restart_capture_test.py \
  -q --no-cov -Werror
```

Tests cover two seed lineages in one merged particle, merger-induced threshold
crossing, capture precedence, irrevocable previous success, unresolved censoring,
real condensation growth, actual CUDA capture, restart before/after merging,
corrupt checkpoint outcome rejection and conflicting ion material maps.
Ruff, formatting and mypy passed (575 source files). These tests establish
bookkeeping and integration; survival-probability convergence is still pending.

## Ion attachment numerics and condensation exclusion

The LF2013 device helper evaluates the published degree-23 polynomials in a
centered logarithmic-radius basis. Coefficients are transformed once using
70-digit decimal arithmetic and uploaded as fp64 values. This is an algebraic
basis change, not a new fit or interpolation. Tests independently evaluate the
original decimal coefficients at nine radii in each valid interval, covering
all 201 aerosol charges and both ion polarities. CPU and CUDA agree with those
references within 1e-10 relative tolerance. Default unsupported rates return a
fault sentinel; optional lower/upper bounds remain explicit. Device helpers
classify free ions, inert neutral products, aerosols and unsupported ion-only
clusters. They are not yet connected to the atmospheric collision scheduler.

`IonCondensationExclusion` gates ion-only proposals before gas-inventory
limiting in every condensation substep. Tests verify exact ion/product masses,
continued uptake by an aerosol carrying ion mass, particle/gas conservation,
isothermal and latent-heat capture/restart, and configuration rejection.
Mixed physical material densities use the CPU surface model's constant Kelvin
density rather than replacing physical densities; a native CPU comparison
passes at 2e-12 relative tolerance. The former mixed-material rejection test
now exercises the still-unsupported nonuniform surface-density case, while
the new parity test covers the newly supported material-density case.

Host command (171 passed, actual CPU/CUDA):

```bash
WARP_CACHE_PATH=/tmp/particula-charged-warp .venv/bin/pytest \
  particula/execution/tests/prepared_ion_condensation_test.py \
  particula/execution/tests/prepared_test.py \
  particula/execution/tests/prepared_cohort_test.py \
  particula/execution/tests/prepared_restart_capture_test.py \
  particula/gpu/dynamics/tests/ion_attachment_test.py \
  particula/dynamics/ions/tests/attachment_test.py \
  -q --no-cov -Werror
```

The direct condensation regression, parallel reductions and autodiff were also
checked after the shared proposal signature changed:

```bash
WARP_CACHE_PATH=/tmp/particula-charged-warp .venv/bin/pytest \
  particula/gpu/kernels/tests/condensation_test.py \
  particula/gpu/kernels/tests/condensation_autodiff_test.py \
  particula/gpu/kernels/tests/condensation_parallel_test.py \
  -q --no-cov -Werror
```

Result: 324 passed, 3 skipped. Ruff, formatting and mypy (581 source files)
passed; the website built with zero link errors.

## Atmospheric collision integration and ion balance

The follow-up to `8d8f476fd` routes ion pairs through the existing sampler and
merge application. Both the exact pair maximum and acceptance probability use
the same derived ion classification and cached LF2013 rates. Validation faults
before collision RNG or physical merge mutation. Neutral products are inert;
same-sign ions do not cluster. Source, cleanup, condensation exclusion and cohort
material contracts are checked together. Derived scratch is recomputed at each
step and does not introduce a new checkpoint payload.

Host CPU/CUDA evidence, 2026-11-14 UTC:

| Command | Result |
| --- | --- |
| `.venv/bin/pytest particula/execution/tests particula/gpu/kernels/tests/brownian_parallel_test.py particula/gpu/dynamics/tests/ion_attachment_test.py -q --no-cov` | 1,721 passed before the final three composition/marker tests |
| `.venv/bin/pytest --cov=particula --cov-report=term-missing --cov-fail-under=80 -Werror` | 10,110 passed, 14 skipped, 1 xfailed; 91.85% coverage; includes all 30 atmospheric tests and actual CUDA |
| `.venv/bin/ruff check particula examples_tests scripts --no-fix` | Passed |
| `.venv/bin/ruff format particula examples_tests scripts --check` | 643 files formatted |
| `.venv/bin/mypy particula --ignore-missing-imports` | Passed, 586 files |
| `.venv/bin/python scripts/docs_examples.py refresh gpu-ion-balance` | Complete RTX 5060 run; figure visually inspected |
| `.venv/bin/python scripts/check_docs_structure.py` | Passed |
| `npm --prefix website test` | 3 passed |
| `npm --prefix website run build` | 453 pages; passed |

The ion-balance example simulates 64 independent boxes per q=0/2/10 condition,
256 slots per box, timestep 1 s and duration 2000 s. A separate sparse master
equation solution supplies the exact finite-population mean and variance;
its truncated upper-state probability is checked below 1e-10. Maximum observed
mean discrepancies were 1.58, 2.67 and 2.54 standard errors, below the fixed six
standard-error acceptance criterion. Net charge was exactly zero. Per-species
mass budgets included source input and neutral-product export, with maximum
relative residual below 1e-14. Observed peak occupied slots were 40, 70 and 128.
The current Python download matches the maintained source and compiles.

This qualifies the source/recombination example, not a survival conclusion.
The 3–50 nm study and its timestep/cleanup/capacity/ensemble convergence remain
outstanding. The production treatment of omitted LF2013 repulsive coefficients
is still awaiting the user's scientific choice; strict rejection is the default.

## Periodic cleanup and documentation reachability follow-up

Hosted documentation CI at `0288c548b` detected that the new ion-balance page
was not linked from authored navigation. The earlier local check ran before
that file was tracked and did not inventory it. Commit `fc18f012d` adds the
owning GPU index and guide links; the committed-file structure check reports
zero errors and its 13 tests pass. This corrects the earlier local-only
structure result rather than treating the failed hosted build as successful.

`IonNeutralCleanup.interval_steps` makes refresh frequency independent of the
physical timestep. A bounded per-box phase counter is checkpointed; cleanup
runs on multiples of the configured interval. Inert neutral products continue
to occupy slots between exports. Unit-count checks still run every call and
invalid checkpoint phases or changed intervals reject. CPU/CUDA cleanup,
atmospheric, cohort and restart regression passed 104 tests, including
midperiod capture/restart and default every-step behavior. Ruff and mypy passed.
The earlier full-package coverage result predates this periodic extension.


The subsequent combined study-helper/ion-exclusion/cleanup/atmospheric/cohort/
restart suite passed 125 tests on host CPU/CUDA. This caught and corrected a
shared-plan assumption: condensation exclusion must not inherit the cleanup
interval parameter. The complete ion-balance example was refreshed again and
its default every-step results retained.

A preliminary neutral survival pilot used the explicit defaults in
`scripts/gpu_studies/lucky_few.py`: 64 independent boxes, one 3 nm seed each,
32 initial 100 nm background particles per box, pulsed vapor replenishment,
and 20,000 s duration. It recorded 2 reached, 62 captured and 0 unresolved
lineages, with maximum mass residual 4.67e-15. The exact 95% success interval
is broad (0.003807–0.108371); this is not a converged production estimate.
The pending timestep/capacity/ensemble work and ion-domain decision remain in
the plan. No ion-effect conclusion follows from this q=0 pilot.


## Neutral growth and survival uncertainty controls

Host CUDA checks at library revision `5aa5b47ff` used the frozen physical
parameters in the survival helper. With background particles disabled, four
seeds per run all reached 50 nm at each of dt = 2, 1 and 0.5 s. First passage
was first observed at 1390 s in all three runs, using 10 s diagnostic spacing.
This satisfies the predeclared crossing-time spread criterion of the larger
of 10 s and 1% of the finest-step crossing time. It does not resolve passage
time within a diagnostic interval. Maximum mass residuals were 2.45e-15,
2.23e-15 and 3.45e-15, respectively.

Survival timestep qualification uses an absolute probability tolerance of
0.01, fixed before the 2048-box comparison. Each run has one initial seed per
independent box. The exact intervals follow [Clopper and Pearson (1934)](https://doi.org/10.1093/biomet/26.4.404).
Two exact 97.5% binomial intervals give at least 95% joint
coverage by the union bound; subtracting their opposite endpoints gives a
conservative interval for the probability difference. This does not assume
independence between runs sharing random streams. The entire difference
interval must lie inside [-0.01, 0.01] to establish the declared tolerance.
An interval that crosses a tolerance boundary is inconclusive, even when
point estimates agree. Any larger confirmatory ensemble must be declared
before its outcomes are examined; pilot estimates alone are not a stopping
rule for nominal confidence coverage.

All six hosted workflows passed on `5aa5b47ff`: test, linting, documentation,
assistant plugins, PyPI packaging and Conda feedstock checks. These are
validation results, not a package release or merge.


The 2048-box pilot gave 39 reached at dt=1 s and 34 at dt=0.5 s, with no
unresolved lineages. The conservative difference interval is
[-0.016076, 0.011208], so the pilot does **not** resolve the declared tolerance.
A fixed confirmation is declared before execution: 8192 independent boxes
per timestep, new root seed 318, dt=1 and 0.5 s, all physical defaults and
20,000 s duration unchanged. Its intervals will be assessed once, separately
from the pilot; neither pooling nor repeated nominal-95% stopping is used.


| Neutral pilot | Reached / 2048 | Maximum mass residual | Synchronized stepping |
| --- | --- | --- | --- |
| dt=1 s, capacity 128 | 39 | 6.22e-15 | 205.7 s |
| dt=0.5 s, capacity 128 | 34 | 5.56e-15 | 407.8 s |
| dt=1 s, capacity 256 | 39 | 6.22e-15 | 304.0 s |

The capacity comparison preserves the complete cohort outcome time series
exactly; maximum observed occupancy was 33 in all runs. These single-run
stepping times exclude preparation, explicit gas forcing and analysis. They
are workload observations on RTX 5060, not a matched CPU speedup or a repeated
performance benchmark. Ion-production capacity and cleanup sensitivity are
still outstanding.


The updated interval helper passed all 21 study-helper tests on host CPU/CUDA
(`.venv/bin/pytest scripts/tests/lucky_few_study_test.py -q -Werror --no-cov`).
Tests cover exact binomial endpoints, sign reversal of difference intervals,
invalid confidence levels and controlled condensation with inventory closure.
Ruff and formatting passed; documentation structure had zero errors and the
website built 453 pages with no link errors. The full-package result above
still predates periodic cleanup; a final complete run remains required.
